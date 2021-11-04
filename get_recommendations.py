from src.utils import *
from src.model import Net
import torch_geometric
import torch


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = prepare_data()
    
    model = Net(train_data.num_features, 128, 64).to(device)
    model.load_state_dict("model.pt")
    
    # give a list of indices to check recommendations for 
    list_to_check = [0, 6, 10]
    
    z = model.encode(test_data.x, test_data.edge_index)
    prob_adj = z @ z.t()
    
    # fill diagonal with 0's so that we don't recommend the orginal paper itself
    prob_adj = prob_adj.fill_diagonal_(0.0)
    prob_adj = prob_adj.detach().cpu()
    sorted, indices = torch.sort(prob_adj[l], descending=True)
    recs = indices[:, :10]