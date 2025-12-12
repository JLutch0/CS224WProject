import torch
from torch import nn, optim
from torch.utils.data import DataLoader

def train_tgn(model, train_dataset, test_dataset=None, epochs=10, batch_size=32, lr=1e-3, device=None):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        model.reset_memory()
        epoch_loss = 0.0

        for batch in train_loader:
            src, dst, t, edge_attr, y, src_static, dst_static, src_dynamic, dst_dynamic = batch

            src = src.to(device)
            dst = dst.to(device)
            t = t.to(device)
            edge_attr = edge_attr.to(device)
            y = y.to(device).unsqueeze(-1)
            src_static = src_static.to(device)
            dst_static = dst_static.to(device)
            src_dynamic = src_dynamic.to(device)
            dst_dynamic = dst_dynamic.to(device)

            optimizer.zero_grad()
            pred = model(src, dst, t, edge_attr, src_static, dst_static, src_dynamic, dst_dynamic)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * src.size(0)
        
        model.eval()
        
        if test_dataset is not None:
            test_loss, test_acc = evaluate_tgn(model, test_loader, criterion, device)
            model.reset_memory()
            train_loss, train_acc = evaluate_tgn(model, train_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        else:
            model.reset_memory()
            train_loss, train_acc = evaluate_tgn(model, train_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    print("Training complete.")

def evaluate_tgn(model, dataloader, criterion, device):
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src, dst, t, edge_attr, y, src_static, dst_static, src_dynamic, dst_dynamic = batch
            
            src = src.to(device)
            dst = dst.to(device)
            t = t.to(device)
            edge_attr = edge_attr.to(device)
            y = y.to(device).unsqueeze(-1)
            src_static = src_static.to(device)
            dst_static = dst_static.to(device)
            src_dynamic = src_dynamic.to(device)
            dst_dynamic = dst_dynamic.to(device)
            
            pred = model(src, dst, t, edge_attr, src_static, dst_static, src_dynamic, dst_dynamic)
            loss = criterion(pred, y)
            
            total_loss += loss.item() * src.size(0)
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_binary == y).sum().item()
            total += y.size(0)
    
    return total_loss / total, correct / total