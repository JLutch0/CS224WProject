import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import NNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

DATA_PATH = "atp_matches_2024.csv"
ENCODERS = {}

def encode(series, name=None):
    # conviententlly, someone else has already written an encoder :)
    if name is None:
        name = series.name

    if name not in ENCODERS:
        le = LabelEncoder()
        series = series.astype(str).fillna("UNK")
        le.fit(series)
        ENCODERS[name] = le
    else:
        le = ENCODERS[name]

    series = series.astype(str).fillna("UNK")
    known_labels = set(le.classes_)
    series = series.apply(lambda x: x if x in known_labels else "UNK")
    return le.transform(series)

def convert_data(df):
    df = df.sort_values("tourney_date")

    timestamps = list(zip(df["tourney_date"], df["round"]))
    edge_index = np.array(list(zip(df["winner_id"], df["loser_id"]))).T

    # match specific detials 
    edge_attrs = np.stack([
        df['surface'],
        df['tourney_level'],
        df['best_of'],
        df['round'],
        df['winner_rank'] - df['loser_rank'],
        df['winner_age'] - df['loser_age']
    ], axis=1)

    return edge_index, edge_attrs, timestamps

def preprocess_data(df):
    """ 
    match_num is an inconsistent count (sometimes it counts down from 300, other times up from 1) of how many matches have been played
    in a tournament, I think I'll get eveyrthing useful from including the round.
    """
    df.drop('match_num', axis=1, inplace=True)
    # draw size shoulden't affect anything
    df.drop('draw_size', axis=1, inplace=True)

    # country shouldent make a difference
    df.drop('winner_ioc', axis=1, inplace=True)
    df.drop('loser_ioc', axis=1, inplace=True)

    # clean up entry data
    df['loser_entry'] = df['loser_entry'].replace('W', 'WC')
    df['loser_entry'] = df['loser_entry'].replace('Alt', 'ALT')
    df['loser_entry'] = df['loser_entry'].replace(np.nan, 'M')

    df['winner_entry'] = df['winner_entry'].replace('W', 'WC')
    df['winner_entry'] = df['winner_entry'].replace('Alt', 'ALT')
    df['winner_entry'] = df['winner_entry'].replace(np.nan, 'M')
    
    # create dictionaries of tournies and players
    tournies = dict(zip(df['tourney_id'], df['tourney_name']))
    players = dict(zip(df['winner_id'], df['winner_name']))
    for my_id, name in zip(df['loser_id'], df['loser_name']):
        players[my_id] = name

    df.drop('tourney_name', axis=1, inplace=True)
    df.drop('winner_name', axis=1, inplace=True)
    df.drop('loser_name', axis=1, inplace=True)
    # dropping minutes for now, may want to reinstate later
    df.drop('minutes', axis=1, inplace=True)

    # Unseeded player's seeds are nan, going to replace with -1
    df['winner_seed'] = df['winner_seed'].replace(np.nan, -1)
    df['loser_seed'] = df['loser_seed'].replace(np.nan, -1)
    
    # lose 130 rows for this, I'm okay with that for now
    df = df.dropna()
    
    categorical_cols = ['surface', 'tourney_level', 'round','winner_entry', 'loser_entry','winner_hand', 'loser_hand']

    for col in categorical_cols:
        df.loc[:, col] = encode(df[col], name=col)

    return df, tournies, players

def build_node_features(df):
    static_cols = ['winner_hand', 'winner_ht', 'loser_hand', 'loser_ht']
    dynamic_cols = ['winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points']

    player_ids = pd.unique(df[['winner_id', 'loser_id']].values.ravel())
    player_ids = np.sort(player_ids)

    node_features_dict = {}

    for _, row in df.iterrows():
        winner_features = np.array([
            row['winner_hand'],
            row['winner_ht'],
            row['winner_rank'],
            row['winner_rank_points']
        ], dtype=float)

        loser_features = np.array([
            row['loser_hand'],
            row['loser_ht'],
            row['loser_rank'],
            row['loser_rank_points']
        ], dtype=float)

        node_features_dict[row['winner_id']] = winner_features
        node_features_dict[row['loser_id']] = loser_features

    all_feats = np.stack(list(node_features_dict.values()))
    mean_feat = np.mean(all_feats, axis=0)

    # I think every player should have this info, but just in case I am using the mean
    for pid in player_ids:
        if pid not in node_features_dict:
            node_features_dict[pid] = mean_feat

    node_feature_matrix = np.stack([node_features_dict[pid] for pid in player_ids])

    return player_ids, node_feature_matrix, node_features_dict

def build_temporal_dataset(edge_index, edge_attrs, node_features, timestamps, player_ids):
    # builds a dataset that shows how the tennis grpah changes with time
    timestamps_tuples = [tuple(ts) for ts in timestamps]

    # sorting first by tournament date, then by round
    unique_times = sorted(list(set(timestamps_tuples)), key=lambda x: (x[0], x[1]))

    edge_indices = []
    edge_attrs_t = []
    node_features_t = []
    targets_t = []

    timestamps_array = np.array(timestamps_tuples)

    for t in unique_times:
        mask = np.all(timestamps_array == t, axis=1)

        if not np.any(mask):
            continue

        edge_indices.append(edge_index[:, mask])
        edge_attrs_t.append(edge_attrs[mask])

        node_features_t.append(node_features)

        targets_t.append(np.ones(edge_index[:, mask].shape[1]))

    dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_attrs_t,
        features=node_features_t,
        targets=targets_t)

    return dataset

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df, tournies, players = preprocess_data(df)
    edge_index, edge_attrs, timestamps = convert_data(df)
    player_ids, node_features, node_dict = build_node_features(df)
    dataset = build_temporal_dataset(edge_index, edge_attrs, node_features, timestamps, player_ids)
