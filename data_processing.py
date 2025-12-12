import pandas as pd
import numpy as np
import torch
from torch import nn
import glob
from TGNModel import TennisTGN
from training import train_tgn
from torch.utils.data import TensorDataset

STAT_COLS = [
    "p1_ace","p1_df","p1_svpt","p1_1stIn","p1_1stWon","p1_2ndWon",
    "p1_SvGms","p1_bpSaved","p1_bpFaced",
    "p0_ace","p0_df","p0_svpt","p0_1stIn","p0_1stWon","p0_2ndWon",
    "p0_SvGms","p0_bpSaved","p0_bpFaced",
]

ROUND_ORDER = {
    "F": 0, "SF": 1, "QF": 2, "R16": 3,
    "R32": 4, "R64": 5, "R128": 6
}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data():
    csv_files = glob.glob("atp_matches_20*.csv") 
    dfs = [pd.read_csv(file) for file in csv_files]
    return pd.concat(dfs, ignore_index=True)

def clean_and_sort_matches(df):
    # These types of matches are so rare, and not like normal tournaments
    df = df[
        ~df["winner_hand"].eq("U") &
        ~df["loser_hand"].eq("U") &
        ~df["round"].isin(["RR", "BR", "ER"])
    ].copy()

    rename_dict = {}
    for col in df.columns:
        if col.startswith("winner_"):
            rename_dict[col] = col.replace("winner_", "p1_")
        elif col.startswith("w_"):
            rename_dict[col] = col.replace("w_", "p1_")
        elif col.startswith("loser_"):
            rename_dict[col] = col.replace("loser_", "p0_")
        elif col.startswith("l_"):
            rename_dict[col] = col.replace("l_", "p0_")
    
    df = df.rename(columns=rename_dict)
    
    # Sort chronologically
    # This is crucial for my entire TGN setup
    df["round_order"] = df["round"].map(ROUND_ORDER)
    df = df.sort_values(["tourney_date", "round_order"]).drop(columns="round_order")
    
    df = df.dropna(subset=STAT_COLS).reset_index(drop=True)

    unknown = set(df["round"].unique()) - set(ROUND_ORDER)
    if unknown:
        print("Warning: Unknown round symbols:", unknown)

    return df

def train_test_split_temporal(df, player2idx, test_date_threshold):
    train_mask = df["tourney_date"] < test_date_threshold
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    train_players = set(player2idx.keys())
    test_valid_mask = (
        df_test["p1_id"].isin(train_players) & 
        df_test["p0_id"].isin(train_players)
    )
    df_test = df_test[test_valid_mask].reset_index(drop=True)
    
    print(f"Train matches: {len(df_train)}")
    print(f"Test matches (before filtering): {len(df[~train_mask])}")
    print(f"Test matches (after filtering): {len(df_test)}")
    print(f"Filtered out {len(df[~train_mask]) - len(df_test)} matches with unseen players")
    
    return df_train, df_test

def build_player_maps(df):
    players = pd.unique(df[["p1_id", "p0_id"]].values.ravel())
    players = [p for p in players if pd.notna(p)]
    player2idx = {p: i for i, p in enumerate(players)}
    idx2player = {i: p for p, i in player2idx.items()}
    return player2idx, idx2player

def build_static_node_features(df, player2idx):
    stacked = pd.DataFrame({
        "player_id": pd.concat([df.p1_id, df.p0_id], ignore_index=True),
        "hand": pd.concat([df.p1_hand, df.p0_hand], ignore_index=True),
        "height": pd.concat([df.p1_ht, df.p0_ht], ignore_index=True),
    }).dropna(subset=["player_id"])

    grouped = stacked.groupby("player_id").agg({
        "hand": lambda s: s.dropna().iloc[0] if s.notna().any() else "R",
        "height": lambda s: s.dropna().iloc[0] if s.notna().any() else 0.0
    })
    grouped = grouped.reindex(list(player2idx.keys())).fillna({"hand": "R", "height": 0.0})

    # Encode handedness
    hand_map = {"R": 0, "L": 1}
    hand_idx = grouped["hand"].map(hand_map).fillna(0).astype(int).to_numpy()
    handed = np.eye(2)[hand_idx]

    # Normalize height. Not sure if nessasary
    height = grouped["height"].to_numpy(dtype=float).reshape(-1, 1)
    mean_ht = np.nanmean(height)
    std_ht = np.nanstd(height) + 1e-6
    height = (height - mean_ht) / std_ht

    static = np.hstack([handed, height])
    static = torch.tensor(static, dtype=torch.float)

    return static

def build_edge_features(df, round_order=ROUND_ORDER, include_stats=True, 
                       surf_map=None, lvl_map=None):
    # Could add other features here, these are just the ones I have
    df_work = df.dropna(subset=STAT_COLS) if include_stats else df

    
    if surf_map is None:
        surf_vals = sorted(df_work.surface.dropna().unique())
        surf_map = {v: i for i, v in enumerate(surf_vals)}
        return_maps = True
    else:
        return_maps = False
    
    surf_idx = df_work.surface.map(surf_map).fillna(0).astype(int).to_numpy()
    surf_oh = np.eye(len(surf_map))[surf_idx]

    if lvl_map is None:
        lvl_vals = sorted(df_work.tourney_level.dropna().unique())
        lvl_map = {v: i for i, v in enumerate(lvl_vals)}
    
    lvl_idx = df_work.tourney_level.map(lvl_map).fillna(0).astype(int).to_numpy()
    lvl_oh = np.eye(len(lvl_map))[lvl_idx]

    round_num = df_work["round"].map(round_order).fillna(-1).to_numpy().reshape(-1, 1)
    
    best_of = (df_work["best_of"] == 5).astype(float).to_numpy().reshape(-1, 1)

    # Pre-match features (known before match)
    pre_match_features = np.hstack([surf_oh, lvl_oh, round_num, best_of])
    
    if include_stats:
        stats = df_work[STAT_COLS].to_numpy(dtype=float)
        stats = np.nan_to_num(stats, nan=0.0)
        result = np.hstack([pre_match_features, stats])
    else:
        result = pre_match_features
    
    if return_maps:
        return result, surf_map, lvl_map
    else:
        return result

def build_tgndataset(df, player2idx, static_feat, surf_map, lvl_map, include_stats=False):
    # The heart of this file
    # Creates the dataset that will get fed into the model.
    num_matches = len(df)

    # All rows currently have p1 as the winner, need to swap half so the model does not 
    # just predict p1 win everytime niavely
    np.random.seed(42)
    swap_mask = np.random.rand(num_matches) < 0.5

    p1_indices = df["p1_id"].map(player2idx).astype(int).to_numpy()
    p0_indices = df["p0_id"].map(player2idx).astype(int).to_numpy()

    src_nodes = np.where(swap_mask, p0_indices, p1_indices)
    dst_nodes = np.where(swap_mask, p1_indices, p0_indices)

    src_nodes_t = torch.tensor(src_nodes, dtype=torch.long)
    dst_nodes_t = torch.tensor(dst_nodes, dtype=torch.long)


    timestamps = torch.tensor(df["tourney_date"].astype(int).to_numpy(), dtype=torch.long)

    edge_features = build_edge_features(df, include_stats=include_stats, 
                                       surf_map=surf_map, lvl_map=lvl_map)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # swapping whether p1 or p0 is flagged as winner or loser
    labels = torch.tensor(~swap_mask, dtype=torch.float)

    p1_static = static_feat[p1_indices]
    p0_static = static_feat[p0_indices]

    # Dynamic features. Future work could be adding more!
    p1_age = torch.tensor(df.p1_age.fillna(0.0).to_numpy(), dtype=torch.float).unsqueeze(1)
    p0_age = torch.tensor(df.p0_age.fillna(0.0).to_numpy(), dtype=torch.float).unsqueeze(1)
    
    p1_rank_points = torch.tensor(df.p1_rank_points.fillna(0.0).to_numpy(), dtype=torch.float).unsqueeze(1)
    p0_rank_points = torch.tensor(df.p0_rank_points.fillna(0.0).to_numpy(), dtype=torch.float).unsqueeze(1)
    
    all_rank_points = torch.cat([p1_rank_points, p0_rank_points])
    mean_points = all_rank_points.mean()
    std_points = all_rank_points.std() + 1e-6
    p1_rank_points = (p1_rank_points - mean_points) / std_points
    p0_rank_points = (p0_rank_points - mean_points) / std_points
    
    p1_dynamic = torch.cat([p1_age, p1_rank_points], dim=1)
    p0_dynamic = torch.cat([p0_age, p0_rank_points], dim=1)

    # Doing the data swapping
    swap_mask_t = torch.tensor(swap_mask, dtype=torch.bool).unsqueeze(1)
    src_static = torch.where(swap_mask_t, p0_static, p1_static)
    dst_static = torch.where(swap_mask_t, p1_static, p0_static)
    src_dynamic = torch.where(swap_mask_t, p0_dynamic, p1_dynamic)
    dst_dynamic = torch.where(swap_mask_t, p1_dynamic, p0_dynamic)

    dataset = TensorDataset(
        src_nodes_t, dst_nodes_t, timestamps, edge_attr, labels,
        src_static, dst_static, src_dynamic, dst_dynamic
    )

    return dataset

if __name__ == "__main__":
    set_seed(42)
    
    df = read_data()
    df = clean_and_sort_matches(df)
    player2idx, idx2player = build_player_maps(df)
    
    # Temporal split. Could be any date
    test_date_threshold = 20240101
    df_train, df_test = train_test_split_temporal(df, player2idx, test_date_threshold)
    
    static_feat = build_static_node_features(df_train, player2idx)
    
    num_players = len(player2idx)
    embed_dim = 32
    learned_emb = nn.Embedding(num_players, embed_dim)
    
    # Build edge features and get encoding maps
    edge_features_sample, surf_map, lvl_map = build_edge_features(df_train, include_stats=False)
    edge_dim = edge_features_sample.shape[1]
    
    print(f"Number of players: {num_players}")
    print(f"Static feature dimension: {static_feat.shape[1]}")
    print(f"Learned embedding dimension: {embed_dim}")
    # Could add to later
    print(f"Dynamic feature dimension: 2 (age + rank_points)")
    print(f"Edge feature dimension: {edge_dim}")
    print(f"Surface categories: {len(surf_map)}")
    print(f"Tournament level categories: {len(lvl_map)}")
    
    train_dataset = build_tgndataset(
        df_train, player2idx, static_feat, surf_map, lvl_map, include_stats=False
    )

    test_dataset = build_tgndataset(
        df_test, player2idx, static_feat, surf_map, lvl_map, include_stats=False
    )
    
    # Model parameters. Seems to do bette wiht higher dims, bottlenecked by gpu vram
    memory_dim = 32
    msg_dim = 32
    node_dim = 64
    out_dim = 1
    static_feat_dim = static_feat.shape[1]
    # age + rank_points
    dynamic_feat_dim = 2  
    
    model = TennisTGN(
        num_nodes=num_players,
        memory_dim=memory_dim,
        msg_dim=msg_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        out_dim=out_dim,
        static_feat_dim=static_feat_dim,
        dynamic_feat_dim=dynamic_feat_dim,
        learned_emb=learned_emb
    )

    train_tgn(model, train_dataset, test_dataset)