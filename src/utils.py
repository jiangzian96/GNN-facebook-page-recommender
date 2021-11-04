from sklearn.metrics import roc_auc_score
import torch_geometric
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import FacebookPagePage

  
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
  
def prepare_data():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	transform = T.Compose([
    T.NormalizeFeatures(),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
	])

	dataset = FacebookPagePage("../data", transform=transform)
	train_data, val_data, test_data = dataset[0]
	
	return train_data.to(device), val_data.to(device), test_data.to(device)
   
  
  
