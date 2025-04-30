import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

def build_loader(trainset,valset,testset,batch_size):
    train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(valset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(testset,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader,test_loader

class my_dataset(Dataset):
    def __init__(self,datas,labels,atts):
        super(my_dataset,self).__init__
        self.datas=datas
        self.labels=labels
        self.atts=atts

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        data=self.datas[index]
        label=self.labels[index]
        att=self.atts[index]
        return data, label, att

def get_data(data_list,train_index,attribute_matrix):
    traindata=[]
    trainlabel=[]
    train_attributelabel=[]
    for item in train_index:
        data=data_list[item]
        traindata.append(data[:,4:10])
        num=data.shape[0]
        trainlabel +=[item]*num
        train_attributelabel +=[attribute_matrix[item,:]]*num
    traindata=np.row_stack(traindata)
    trainlabel=np.row_stack(trainlabel)
    train_attributelabel=np.row_stack(train_attributelabel)
    return traindata,trainlabel,train_attributelabel

def create_data(df,test_index=[3,4],train_ration=0.5):
    
    fault0=df[df['fault_type']=='0000']
    fault1=df[df['fault_type']=='1001']
    fault2=df[df['fault_type']=='0110']
    fault3=df[df['fault_type']=='1011']
    fault4=df[df['fault_type']=='0111']
    fault5=df[df['fault_type']=='1111']


    train_index=list(set(np.arange(6))-set(test_index))
    attribute_matrix=np.array([[0,0,0,0],
                            [1,0,0,1],
                            [0,1,1,0],
                            [1,0,1,1],
                            [0,1,1,1],
                            [1,1,1,1]])

    data_list=[fault0.values.astype(np.float64),fault1.values.astype(np.float64),fault2.values.astype(np.float64),
                fault3.values.astype(np.float64),fault4.values.astype(np.float64),fault5.values.astype(np.float64)]

    train_index.sort()
    test_index.sort()
    train_attributematrix=attribute_matrix[train_index]
    test_attributematrix=attribute_matrix[test_index]
    seendata,seenlabel,seen_attributelabel = get_data(data_list,train_index,attribute_matrix)
    num_seen = seendata.shape[0]
    split = np.random.choice(num_seen,num_seen,replace=False)
    train_index = int(train_ration*num_seen)
    traindata,trainlabel,train_attributelabel = seendata[split[:train_index]],seenlabel[split[:train_index]],seen_attributelabel[split[:train_index]]
    valdata,vallabel,val_attributelabel = seendata[split[train_index:]],seenlabel[split[train_index:]],seen_attributelabel[split[train_index:]]
    testdata,testlabel,test_attributelabel=get_data(data_list,test_index,attribute_matrix)


    return  traindata,trainlabel.squeeze(),train_attributelabel,train_attributematrix,\
            valdata,vallabel.squeeze(),val_attributelabel,\
            testdata,testlabel.squeeze(),test_attributelabel,test_attributematrix,attribute_matrix
 



