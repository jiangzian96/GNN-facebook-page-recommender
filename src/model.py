class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_head = 8
        self.out_head = 1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*self.in_head, out_channels, heads=self.out_head, concat=False, dropout=0.6)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
