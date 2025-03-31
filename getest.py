import torch
import torch.nn.functional as F
# from torch_geometric.datasets import Planetoid
from TAGLAS.datasets import Cora, Pubmed, WikiCS, Products, Arxiv
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges

# 加载数据集
cora = Cora()
pubmed = Pubmed()
wikics = WikiCS()
products = Products()
arxiv = Arxiv()
datasets = [cora, pubmed, arxiv, products, wikics]

xc = torch.load('TAGDataset/cora/task/ST/node_features.pt')
xpu = torch.load('TAGDataset/pubmed/task/ST/node_features.pt')
xa = torch.load('TAGDataset/arxiv/task/ST/node_features.pt')
xpr = torch.load('TAGDataset/products/task/ST/node_features.pt')
xw = torch.load('TAGDataset/wikics/task/ST/node_features.pt')

xs = [xc, xpu, xa, xpr, xw]

# # 划分训练集和测试集的边
# data = train_test_split_edges(data)


# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

accs = []
epochs = [30, 30, 80, 50, 30]
for i in range(len(datasets)):

    # 初始化模型、优化器和损失函数
    dataset = datasets[i]
    device = 'cuda:0'
    model = GCN(dataset.num_node_features, 64, 32).to(device)
    data = dataset[0]
    data = data.to(device)
    print(data)
    x = xs[i].to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


    # 定义 K 值
    K = 10 * dataset.num_classes


    # 提取 K 条正边用于训练
    train_pos_edge_index = data.edge_index[:, :K]
    test_pos_edge_index = data.edge_index[:, K:]

    # 生成 K 条负边用于训练
    train_neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=K, method='sparse')

    # 训练模型
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        logits = model.decode(z, train_pos_edge_index, train_neg_edge_index)
        labels = torch.cat([torch.ones(train_pos_edge_index.size(1)),
                            torch.zeros(train_neg_edge_index.size(1))]).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    # 测试模型
    def test():
        model.eval()
        z = model.encode(x, data.edge_index)
        pos_edge_index = test_pos_edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1), method='sparse')
        logits = model.decode(z, pos_edge_index, neg_edge_index)
        labels = torch.cat([torch.ones(pos_edge_index.size(1)),
                            torch.zeros(neg_edge_index.size(1))]).to(device)
        preds = (logits > 0).float()
        acc = (preds == labels).sum().item() / labels.size(0)
        return acc

    # 训练 200 个 epoch
    for epoch in range(1, 1+epochs[i]):
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    accs.append(acc)

print(accs)
        