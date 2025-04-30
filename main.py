#%%
import os

import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch.optim as optim
from read_data import create_data,my_dataset,build_loader
from mae import MAE
from config import _C as cfg
from classification import our_eval
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import torch.nn.functional as F
import susi

%matplotlib inline
#%%
np.random.seed(904)
torch.manual_seed(904)
torch.cuda.manual_seed(904)


parser = argparse.ArgumentParser(description='MCA')
parser.add_argument('--config',default='mca.yaml',type=str,help='network setting')
args = parser.parse_args(args=[])

cfg.merge_from_file(args.config)
# cfg.freeze()

if cfg.TRAIN.device=='gpu':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

row=cfg.SOM.row
col=cfg.SOM.col
som_epoch=cfg.TRAIN.som_epoch
mae_epoch=cfg.TRAIN.mae_epoch
lr = cfg.TRAIN.lr
batch_size = cfg.TRAIN.batch_size
datapath=cfg.DATA.path
modes=[cfg.EVALUATION.classifier]
df = pd.read_csv(datapath)
df['fault_type'] = df['G'].astype('str') + df['C'].astype('str') + df['B'].astype('str') + df['A'].astype('str')
traindatas,trainlabels,trainatts, \
att_s,valdatas,vallabels,valatts,\
testdatas,testlabels,testatts,att_u,atts=create_data(df,cfg.DATA.test_index,train_ration=0.7)
#%%
if cfg.DATA.datatype=='float32':
    traindatas,valdatas,testdatas = traindatas.astype(np.float32),valdatas.astype(np.float32),testdatas.astype(np.float32)
else:
    traindatas,valdatas,testdatas = traindatas.astype(np.float64),valdatas.astype(np.float64),testdatas.astype(np.float64)

train,val,test = traindatas,valdatas,testdatas

if cfg.DATA.scale:
    scale = StandardScaler()
    train = scale.fit_transform(train)
    val = scale.transform(val)
    test = scale.transform(test)

train_x, val_x,test_x = train,val,test
#%%
Traindatas, Valdatas, Testdatas = torch.from_numpy(train).to(device), torch.from_numpy(val).to(device), torch.from_numpy(test).to(device)
trainset = my_dataset(Traindatas,torch.from_numpy(trainlabels).to(device),torch.from_numpy(trainatts).to(device))
valset = my_dataset(Valdatas,torch.from_numpy(vallabels).to(device),torch.from_numpy(valatts).to(device))
testset = my_dataset(Testdatas,torch.from_numpy(testlabels).to(device),torch.from_numpy(testatts).to(device))     
trainloader, valloader, testloader = build_loader(trainset,valset,testset,batch_size=batch_size) 

losses = list()
#%%
print('*************************Train SOM*******************************')
n_samples = train.shape[0]
row,col = int(np.sqrt(5*np.sqrt(n_samples)))+1,int(np.sqrt(5*np.sqrt(n_samples)))+1

som = susi.SOMClassifier(row,col,n_iter_unsupervised=50000,n_iter_supervised=50000,\
            nbh_dist_weight_mode='pseudo-gaussian',do_class_weighting=False,\
            n_jobs=1,verbose=True)
som.fit(train,trainlabels.astype(np.int64))
memory = som.unsuper_som_.reshape(-1,train.shape[1]).astype(np.float32)
memory = torch.from_numpy(memory).to(device)

#%%
print('Building Model...')
cfg.ENCODER.num_node = [train.shape[1],train.shape[1],memory.shape[1]]
cfg.DECODER.num_node = [memory.shape[1],6]
cfg.SOM.input_size = memory.shape[1]
model = MAE(cfg,device).to(device)
optimizer = optim.SGD(model.parameters(),lr=lr)
loss_fc = nn.MSELoss()

y_true = np.r_[np.zeros(vallabels.shape[0]),np.ones(testlabels.shape[0])]
y_label = np.r_[vallabels,testlabels]
print('*************************Train MAE*******************************')
# def main():
best_h = 0
best_result = {}
best_result['y_mark'] = y_true
best_result['y_label'] = y_label
best_result['train_label'] = trainlabels
best_result['val_label'] = vallabels
best_result['test_label'] = testlabels
best_result['memory'] = memory.detach().cpu().numpy()
hs = []
ss = []
us = []
mes = []
res = []
lss = []

for epoch in range(mae_epoch):
    losses = 0
    re_losses = 0
    me_losses = 0
    model.train()
    print('Epoch:{}'.format(epoch,end=''))
    for idx, (x, y, att) in enumerate(trainloader):   
        optimizer.zero_grad()
        z, z_, output,att_weight = model(x,memory)
        re_loss = model.re_loss(x,output)
        me_loss,c = model.memory_loss(z,z_)
        me_loss = cfg.MAE.lamd_c *me_loss
        # loss = re_loss
        loss = re_loss + me_loss
        loss.backward()
        optimizer.step()

        losses += loss.item()
        re_losses += re_loss.item()
        me_losses += me_loss.item()

    losses = losses / len(trainloader)
    re_losses = re_losses / len(trainloader)
    me_losses = me_losses / len(trainloader)
    mes.append(me_losses)
    res.append(re_losses)
    lss.append(losses)

    model.eval()
    with torch.no_grad():
        _,_, _,train_att_weight = model(trainset.datas,memory)
        _,seen_z, seen_output,seen_att_weight = model(valset.datas,memory)
        seen_re_loss = torch.norm((valset.datas - seen_output),dim=1)

        _,unseen_z, unseen_output,unseen_att_weight = model(testset.datas,memory)
        unseen_re_loss = torch.norm((testset.datas-unseen_output),dim=1)

        
    print('loss:{:.4f}, re_loss:{:.4f}, me_loss:{:.4f}, sl:{:.4f}, unl:{:.4f}'\
    .format(losses, re_losses, me_losses, seen_re_loss.mean().item(),unseen_re_loss.mean().item()))
    seen_error, unseen_error = seen_re_loss.detach().cpu().numpy(), unseen_re_loss.detach().cpu().numpy()
    y_scores = np.r_[seen_error, unseen_error]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if cfg.EVALUATION.metric=='auc':
        opt_threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        opt_threshold = thresholds[np.argmax((precision*recall)/(precision+recall+1e-12))]
    auc_area = auc(fpr,tpr)
    seen_marks =  (np.array(seen_error<opt_threshold)+np.array(seen_error==opt_threshold)).astype(int)
    unseen_marks = np.array(unseen_error>opt_threshold).astype(int)
    acc, u,s,h,train_z,val_z,test_z,y_pre_label = our_eval(cfg,model,Traindatas,train_x,trainatts,trainlabels,Valdatas,val_x,seen_marks,vallabels,Testdatas,test_x,unseen_marks,testlabels,att_s, att_u, atts)
    hs.append(h)
    ss.append(s)
    us.append(u)
    print('auc:',auc_area)

    if ~(h<best_h):
        best_h = h
        best_result['best_epoch'] = epoch
        best_result['fpr'] = fpr
        best_result['tpr'] = tpr
        best_result['thresholds'] = thresholds
        best_result['seen_error'] = seen_error      
        best_result['unseen_error'] = unseen_error      
        best_result['opt_threshold'] = opt_threshold
        best_result['auc'] = auc_area
        best_result['acc'] = acc
        best_result['u'] = u
        best_result['s'] = s
        best_result['h'] = h
        best_result['c'] = c.detach().cpu().numpy()
    # print('best epoch:{}, best_h:{:.4f}'.format(best_result['best_epoch'],best_result['h']))
    print('The best epoch: {}'.format(best_result['best_epoch']))
    print('Acc: {:.4f}, S: {:.4f}, U: {:.4f}, H: {:.4f},auc:{:.4f}'.format(best_result['acc'],best_result['s'],
                                                            best_result['u'],best_result['h'], best_result['auc']))
    print('\n')
  

#%%
plt.figure()
plt.figure()
plt.plot(best_result['fpr'],best_result['tpr'])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC of MCA')

Re_losses = np.r_[best_result['seen_error'],best_result['unseen_error']]
Labels = np.r_[vallabels,testlabels]
Re_losses=Re_losses[np.argsort(Labels)]
Labels.sort()
N =[j+1 for j in range(Re_losses.shape[0])]
plt.scatter(N,Re_losses,c=Labels)
# %%
