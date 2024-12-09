from .gpse import GPSE, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPSE_MLP(nn.Module):

    def __init__(self, gpse, dim_in, dim_out, dim_hid, num_layers):
        super(GPSE_MLP, self).__init__()
        self.gpse = gpse if gpse is not None else GPSE.from_pretrained('molpcba')
        self.mlp = MLP(dim_in, dim_out, dim_hid, num_layers)

    def forward(self, batch):
        b = batch.__copy__()
        b, _ = self.gpse(b)
        return self.mlp(b)

    def constractive_loss(self, batch, output, t=0.2):
        num_nodes = len(batch.x)
        A = torch.zeros((num_nodes, num_nodes))  
        A[batch.edge_index[0], batch.edge_index[1]] = 1  
        A[batch.edge_index[1], batch.edge_index[0]] = 1
        f = lambda x: torch.exp(x/t) 
        loss = 0
        def sim(z1, z2):
            z1 = F.normalize(z1) 
            z2 = F.normalize(z2) 
            return torch.mm(z1, z2.t()) 
        sim_m = f(sim(output, output))
        neg_sim = (sim_m.sum(0) - sim_m.diag()).sum()
        pos_sim = torch.sum(sim_m[A==1])
        loss = -torch.log(pos_sim/neg_sim)
        return loss/len(output)