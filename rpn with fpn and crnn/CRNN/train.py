import torch
import torch.cuda
import torch.nn as nn
import dataset
from dataset import MyDataset
import converter
from torch.utils.data import DataLoader
import os
import random
import json
import model
import torch.nn.functional as F

hyperparameters = {
    "lr": 0.0001,
    "batch_size": 100,
    "num_epochs": 10,
    "manualSeed":1234,
    "alphabet":'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    "nhidden":256,
    "pretrained":'CRNN/netCRNN.pth',
    "optimizer":"Adam",
    "beta1":0.5,
    "weight_decay":0.1
}
file_path = 'hyperparameters.json'
with open(file_path, 'w') as json_file:
    json.dump(hyperparameters, json_file, indent=4)
with open(file_path, 'r') as json_file:
    hyperparameters = json.load(json_file)



train_dataset = MyDataset(0)
test_dataset = MyDataset(1)
val_dataset = MyDataset(2)
train_loader = DataLoader(train_dataset,batch_size=hyperparameters["batch_size"],shuffle=True,drop_last=True,pin_memory=True)
test_dataset = DataLoader(test_dataset,batch_size=hyperparameters["batch_size"],shuffle=True,drop_last=True,pin_memory=True)
val_dataset = DataLoader(val_dataset,batch_size=hyperparameters["batch_size"],shuffle=True,drop_last=True,pin_memory=True)

nclass = len(hyperparameters["alphabet"])+1
converter = converter.strLabelConverter(hyperparameters["alphabet"])
criterion = nn.CTCLoss(blank= 0,reduction='mean')
print(converter.alphabet)
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.0)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(0.0, 0.0)
#         m.bias.data.fill_(0)
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        # 使用 Xavier 初始化方法初始化卷积层权重
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)  # 可选：初始化偏置为 0
    elif isinstance(m, nn.BatchNorm2d):
        # 使用正态分布初始化 BatchNorm 层权重和偏置
        nn.init.normal_(m.weight, mean=0, std=1.0)
        nn.init.constant_(m.bias, 0.0)

model = model.CRNN(imgH=32,nclass=nclass,nh=hyperparameters["nhidden"],nc=1)

#model.apply(weights_init)

if hyperparameters["pretrained"] != '':
    print('loading pretrained model from %s' %  hyperparameters["pretrained"])
    model.load_state_dict(torch.load(hyperparameters["pretrained"]))
print(model)

if hyperparameters["optimizer"]=="Adam":
    optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameters["lr"],betas=(hyperparameters["beta1"],0.999),weight_decay=hyperparameters["weight_decay"])
elif hyperparameters["optimizer"]=="SGD":
    optimizer = torch.optim.SGD(model.parameters(),lr = hyperparameters["lr"])
else: optimizer = torch.optim.Adadelta(model.parameters(),lr = hyperparameters["lr"])


class Trainer():
    def __init__(self,model,criterion,optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                                              gamma=0.65)
    def train(self,epoch):
        i = 0
        j = 1
        while j:
            for data in train_loader:
                i = i+1
                if i>epoch:
                    j = j-1
                    break

                image,label = data
                #print(label)
                text,length = converter.encode(label)
                text_tensor = torch.tensor(text).reshape(-1,6)
                preds  = self.model(image)
               
                preds = F.log_softmax(preds, dim=2)
                input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)  # 假设输入序列长度相同，为 preds 的时间步数
                target_lengths = torch.full((text_tensor.size(0),), text_tensor.size(1), dtype=torch.long)  # 假设目标序列长度相同，为 text 的字符数
                #print(f'pred{preds},text_tensor{text_tensor},inputleng{input_lengths},target{target_lengths}')
                print(f'{preds.max(2)[1]}')
                loss = self.criterion(preds, text_tensor,input_lengths,target_lengths)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                if i %1==0:print(f'epoch:{i},loss:{loss}')

                if i %10 == 0: 
                    torch.save(self.model.state_dict(), 'saved_model.pth')
                    print("Your model has been saved")


if __name__ == '__main__':
    Trainer = Trainer(model = model,criterion=criterion,optimizer=optimizer)
    Trainer.train(1000)