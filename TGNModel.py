import torch
from torch import nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import TimeEncoder, LastAggregator

class ECCMessage(nn.Module):
    def __init__(self, edge_dim, memory_dim, msg_dim):
        super().__init__()
        self.edge_network = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * memory_dim * msg_dim)
        )
        self.memory_dim = memory_dim
        self.msg_dim = msg_dim
        self.out_channels = msg_dim

    def forward(self, src_mem, dst_mem, raw_msg, t_enc):
        W = self.edge_network(raw_msg).reshape(-1, 2 * self.memory_dim, self.msg_dim)
        x = torch.cat([src_mem, dst_mem], dim=-1).unsqueeze(1)
        msg = torch.bmm(x, W).squeeze(1)
        return msg


class TennisTGN(nn.Module):
    def __init__(self, num_nodes, memory_dim, msg_dim, node_dim,
                 edge_dim, out_dim, static_feat_dim, dynamic_feat_dim, learned_emb, time_dim=16):
        super().__init__()
        
        self.learned_emb = learned_emb
        self.static_feat_dim = static_feat_dim
        
        self.time_encoder = TimeEncoder(time_dim)

        raw_msg_dim = edge_dim + time_dim 
        self.message_module = ECCMessage(raw_msg_dim, memory_dim, msg_dim)

        # This means we are using the last edge added
        # i.e we are updating one match at a time
        self.aggregator_module = LastAggregator()

        # How node memory is intalized
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=self.message_module,
            aggregator_module=self.aggregator_module
        )

        # memory + static + dynamic + learned_emb 
        node_input_dim = memory_dim + static_feat_dim + dynamic_feat_dim + learned_emb.embedding_dim 
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, out_dim)
        )

    def forward(self, src, dst, t, edge_attr, src_static, dst_static, src_dynamic, dst_dynamic):
        time_enc = self.time_encoder(t.float().unsqueeze(-1))

        # Note time is getting rolled in with edge attributes
        raw_msg = torch.cat([edge_attr, time_enc], dim=-1)

        # Update memory with edge attributes and time
        self.memory.update_state(src, dst, t, raw_msg)

        src_mem, _ = self.memory(src)
        dst_mem, _ = self.memory(dst)

        # We have to detach so the model does not backprop on itself twice
        src_mem = src_mem.detach()
        dst_mem = dst_mem.detach()

        # get learned embeddins of each node
        src_learned = self.learned_emb(src)
        dst_learned = self.learned_emb(dst)

        src_full = torch.cat([src_mem, src_static, src_learned, src_dynamic], dim=-1)
        dst_full = torch.cat([dst_mem, dst_static, dst_learned, dst_dynamic], dim=-1)

        # get our final node embedings
        src_embed = self.node_mlp(src_full)
        dst_embed = self.node_mlp(dst_full)

        # Make a prediction based on new node embedings and edge attributes
        pred = self.predictor(torch.cat([src_embed, dst_embed, edge_attr], dim=-1))
        return pred

    def reset_memory(self):
        # To clear memory of every node between training epochs
        self.memory.reset_state()