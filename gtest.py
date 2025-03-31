import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from TAGLAS.datasets import Cora, Pubmed, WikiCS, Products, Arxiv

# 加载数据集
cora = Cora()
pubmed = Pubmed()
wikics = WikiCS()
products = Products()
arxiv = Arxiv()
datasets = [cora, pubmed, arxiv, products, wikics]

# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义 GAT 模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义 GAT 模型
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

for dataset in datasets:

    data = dataset[0]
    print(data)

    # 定义 10-shot 训练数据
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for c in range(dataset.num_classes):
        class_indices = (data.label_map == c).nonzero(as_tuple=True)[0]
        train_mask[class_indices[:10]] = True

    test_mask = ~ train_mask

    # 初始化模型、优化器和损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(dataset.num_node_features, 512, dataset.num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 训练模型
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x_original, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.label_map[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    # 测试模型
    def test():
        model.eval()
        out = model(data.x_original, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[test_mask] == data.label_map[test_mask]
        test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc

    # 训练过程
    for epoch in range(200):
        loss = train()
        if (epoch + 1) % 10 == 0:
            test_acc = test()
            print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
