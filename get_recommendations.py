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
    indices_to_check = [0, 6, 10]
    recs = get_recommendations(test_data, indices_to_check)
    print(recs)
