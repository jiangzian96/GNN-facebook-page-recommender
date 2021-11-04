from src.utils import *
from src.model import Net
from sklearn.metrics import roc_auc_score
import torch_geometric
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import FacebookPagePage

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = prepare_data()
    
    model = Net(train_data.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    
    for epoch in range(1, 100):
        model.train()
        
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse').to(device)

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val = val_auc
            final_test_auc = test_auc
            torch.save(model.state_dict(), "model.pt")
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')