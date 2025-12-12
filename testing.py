import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(5, 16, device=device)
edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long, device=device)
data = Data(x=x, edge_index=edge_index)

model = GCNConv(16, 8).to(device)
out = model(data.x, data.edge_index)
print(out)