import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f"weight{i}")


class LinesLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.bias1 = nn.Parameter(torch.zeros_like(self.bias))
        # torch.nn.init.ones_(self.weight1)
        torch.nn.init.xavier_normal( self.weight1 )
        torch.nn.init.zeros_(self.bias1)

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


    def forward(self, x, subspace = None):
        if subspace == None:
            w, b = self.get_weight()
            # print( "subspace:", subspace )
        elif subspace == 0:
            w, b = self.weight, self.bias
            # print( "subspace:", subspace )
        else:
            w, b = getattr(self, f"weight{subspace}"), getattr(self, f"bias{subspace}")
            # print( "subspace:", subspace )

        x =  F.linear(x, w, b)
        return x
        



class Subspace_MLP(nn.Module):  # pretrain the classifier to make income predictions.

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Subspace_MLP, self).__init__()
        self.lin1 = LinesLinear(n_features, n_hidden)
        self.lin2 = LinesLinear(n_hidden, n_hidden)
        self.lin3 = LinesLinear(n_hidden, n_hidden)
        self.lin4 = LinesLinear(n_hidden, 1)

        self.layer_norm = nn.LayerNorm(n_hidden, elementwise_affine=False)


    def forward(self, x, subspace = None):
        # x = self.network(x)
        x = self.lin1(x, subspace = subspace)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x, subspace = subspace)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin3(x, subspace = subspace)
        # h = F.relu( x )
        h = torch.sigmoid(x)
        # h = self.layer_norm(h)
        # h1 = F.normalize(h, p=2)

        x = self.lin4(h, subspace = subspace)
        x = torch.sigmoid(x)

        # return h1, x
        return x
