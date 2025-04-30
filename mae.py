#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, cfg):
        super(encoder,self).__init__()

        self.input_size = cfg.SOM.input_size
        self.num_node = cfg.ENCODER.num_node
        self.num_layer = len(self.num_node)
        
        self.layers=[]
        for l in range(self.num_layer-1):
            self.layers.append(nn.Linear(self.num_node[l],self.num_node[l+1]))
            self.layers.append(nn.BatchNorm1d(self.num_node[l+1]))
            # if l!=(self.num_layer-2):
            self.layers.append(nn.LeakyReLU(0.2,inplace=True))

        self.encoder_layers=nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.encoder_layers(x)
        return x

class decoder(nn.Module):
    def __init__(self, cfg):
        super(decoder,self).__init__()

        self.input_size = cfg.SOM.input_size
        self.num_node = cfg.DECODER.num_node
        self.num_layer = len(self.num_node)
        
        
        self.layers=[]
        for l in range(self.num_layer-1):
            self.layers.append(nn.Linear(self.num_node[l],self.num_node[l+1]))
            self.layers.append(nn.BatchNorm1d(self.num_node[l+1]))
            if l!=(self.num_layer-2):
                self.layers.append(nn.ReLU(0.2,inplace=True))

        self.decoder_layers=nn.Sequential(*self.layers)


    def forward(self,x):
        x=self.decoder_layers(x)
        return x



class MAE(nn.Module):
    def __init__(self,cfg,device):
        super(MAE,self).__init__()
        self.cfg = cfg
        self.encoder = encoder(cfg)
        self.decoder =  decoder(cfg)
        # self.weight=nn.Parameter(torch.tensor(encoder_cfg.num_node,encoder_cfg.out_size[-1]))
        self.shrink = cfg.MAE.shrink
        self.similarity = cfg.MAE.similarity
        self.shrink_thres = cfg.MAE.shrink_thres
        self.lamd_off = cfg.MAE.lamd_off
        self.lamd_c = cfg.MAE.lamd_c
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineSimilarity()
        self.me_fc = nn.Linear(cfg.SOM.input_size,cfg.ENCODER.num_node[-1])
        self.bn = nn.BatchNorm1d(cfg.DECODER.num_node[0],affine=False)
        self.W = nn.Parameter(torch.randn(cfg.ENCODER.num_node[-1],cfg.ENCODER.num_node[-1]), requires_grad=True)

    def hard_shrink_relu(self,input, lambd=0, epsilon=1e-12):
        output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def som_mapping(self,x, current_iter, max_iter):
        som_loss, mae_x = self.som.self_organizing(x, current_iter, max_iter)
        return  som_loss, mae_x

            
    def re_loss(self,x,output):
        
        if self.cfg.MAE.re_loss =='mse':
            re_loss = F.mse_loss(x,output)
            return re_loss
        else:
            re_loss = 1 - self.cos_loss(x,output)
            return re_loss.mean() 


    def memory_loss(self,z,z_):
        N, D = z.shape
        c = (self.bn(z).T @ self.bn(z_)) / N 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + (self.lamd_off * off_diag)
        return loss,c

    def get_loss(self,re_loss,me_loss):
        loss = re_loss + self.lamd_c * me_loss
        return loss

    def forward(self,x,memory):
        z = self.encoder(x)
        if self.similarity=='cos':
            att_weight = F.linear(z,memory)
            att_weight = F.softmax(att_weight,dim=1)
        elif self.similarity=='Euclidean':
            att_weight = torch.cdist(z,memory,p=2)
            att_weight = F.softmax(-att_weight,dim=1)
        else:
            att_weight = torch.einsum('ij,jj,jk->ik',z,self.W,memory.T)
            att_weight = F.softmax(att_weight,dim=1)
       
        att_weight = F.normalize(att_weight,p=1,dim=1)

        z_ = F.linear(att_weight,memory.T)
        output = self.decoder(z_)


        return z,z_,output,att_weight
    


