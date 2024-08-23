import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.sgd
from dataclasses import dataclass

import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from torch.profiler import profile, record_function, ProfilerActivity


start_token = '<s>'
end_token = '<e>'
null_token = '<0>'
name_tensor_length_max = 12

class NamesToData():
    def __init__(self,dataset,names):
        X, Y = [], []
        for name in names:
            name = [c for c in name]
            # name = [start_token] + name + [end_token]
            # #padding = ( (name_tensor_length_max + 1 )// len(name) ) + 1
            # # #name = name * padding
            
            # x = name[:name_tensor_length_max]
            # y = name[1:name_tensor_length_max + 1]
            x = [start_token] + name
            y = name + [end_token]
            print(x)
            x = [dataset.tok2int[c] for c in x]
            y = [dataset.tok2int[c] for c in y]
            
            
            X.append(x)
            Y.append(y)
        self.x = torch.tensor(X).to(dataset.device)
        self.y = torch.tensor(Y).to(dataset.device)
        self.names = names

class Dataset():
    def __init__(self,names,device,train_split = 0.8,dev_split = 0.1):
        self.device = device
        chars = sorted(list(set(''.join(names))))
        chars.insert(0,start_token)
        chars.insert(1,end_token)
        chars.insert(2,null_token)
        self.int2tok = {i:chars[i] for i in range(len(chars))}
        self.tok2int = {c:i for i,c in self.int2tok.items()}
        self.num_chars = len(self.int2tok)

        random.seed(1)
        random.shuffle(names)
        n1 = int(len(names)*train_split)
        n2 = int(len(names)*dev_split) + n1

        self.train = NamesToData(self,names[:n1])
        self.dev = NamesToData(self,names[n1:n2])
        self.test = NamesToData(self,names[n2:])

class Trainer():
    def __init__(self,hps,model,dataset):
            self.hps:HyperParams = hps
            self.model:nn.Module = model
            self.dataset = dataset
            self.writer = SummaryWriter(comment=hps.__str__())
            self.optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr,weight_decay=hps.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=hps.lr_step,gamma=hps.lr_decay)


    def do_epoch(self,epoch):
        for X_batch, Y_batch in self.loader_train:
                X_batch, Y_batch = X_batch.to(self.hps.device), Y_batch.to(self.hps.device)

                loss = self.model.loss(X_batch,Y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss/train',loss,epoch)


    def train(self,scoring=False,saving = False):
        time_start = time.time()
        self.loader_train = data.DataLoader(data.TensorDataset(self.dataset.train.x, self.dataset.train.y),
                                            shuffle=True, batch_size=self.hps.batch_size)
        best_devloss = torch.inf
        for epoch in range(self.hps.epochs):
            self.do_epoch(epoch)
            devloss = self.model.loss(self.dataset.dev)
            self.writer.add_scalar('loss/dev',devloss,epoch)
            print(f'epoch {epoch:03d}, loss {devloss:.5f},',end='')
            if scoring:
                score = self.model.test_model(self.hps.testing_num)
                self.writer.add_scalar('score',score,epoch)
                print(f' score {score:.5f}',end='')
            if devloss < best_devloss:
                best_devloss = devloss
                best_epoch = epoch
                if saving:
                    self.model.save('checkpoint')
            print(f' : best loss of {best_devloss:2.5f} at epoch {best_epoch:03d}',end='')
            print(f'\r',end='')
            self.scheduler.step()

        self.writer.flush()
        time_elapsed = time.time() - time_start
        print(f'\ntime elapsed: {time_elapsed:.1f}s')
        return

class RNN(nn.Module):
        def __init__(self,dataset,hps):
            super().__init__()
            self.dataset = dataset
            self.hps = hps
            

            self.embed = nn.Embedding(dataset.num_chars,self.hps.emb_dim)
            self.lstm = nn.LSTM(input_size  =   self.hps.emb_dim,
                                hidden_size =   self.hps.hidden_size,
                                num_layers  =   self.hps.num_layers,
                                batch_first =   True,
                                dropout     =   self.hps.dropout_rate)
            self.relu = nn.LeakyReLU()
            self.linear = nn.Linear(self.hps.hidden_size,dataset.num_chars)
            super().to(self.hps.device)

        def forward(self,x):

            x = self.embed(x.long())
            x, _ = self.lstm(x)
            x = self.linear(x)

            return x
        
        def forward_step(self,y):
            x = y
            x = self.embed(x.long())
            x, _ = self.lstm(x)
            x = self.linear(x)
            probs = F.softmax(x,dim=2)[:,-1,:]
            #new_char = torch.amax(probs,dim=1).to(device=self.hps.device).view(-1,1)
            new_char = torch.multinomial(probs,num_samples=1).to(device=self.hps.device)
            y = torch.cat((y,new_char),dim=1)
            #print(''.join([self.dataset.int2tok[i.item()] for i in y[0] if i.item() not in [0,1,2]]))
            return y
        
        def loss(self,*args):
            if isinstance(args[0],NamesToData):
                data = args[0]
                x, y = data.x, data.y
            else:
                x,y = args

            logits = self.forward(x)
            y = F.one_hot(y,self.dataset.num_chars)
            
            loss = F.cross_entropy(logits,y.float())
            return loss

        def save(self,filename):
            torch.save(self.state_dict(), filename + ".pth")
            with open(filename + '.pkl', 'wb') as f: 
                pickle.dump((self.dataset,self.hps), f)
        
        def load(filename):
            with open(filename + '.pkl','rb') as f:
                dataset,hps = pickle.load(f)
            m = RNN(dataset,hps)
            m.load_state_dict(torch.load(filename + ".pth"))
            return m

        def generate_name(self,name_start='',batch_size = None):
            if batch_size == None:
                batch_size = self.hps.batch_size
            name = name_start
            context = [start_token] + [c for c in name]
            context = [[self.dataset.tok2int[i] for i in context]]

            context = torch.tensor(context,device=self.hps.device)
            x = context.expand(batch_size,-1)
            for char_i in range(name_tensor_length_max - 1):
                x = self.forward_step(x)

            output = []
            for row in x:
                name = ''.join([self.dataset.int2tok[i.item()] for i in row if i.item() not in [0,1,2]])
                #name = name.split(end_token)[0]
                output.append(name)
            return output

        def test_model(self,number_of_names = 1e3,name_start = '',print_names = False):
            score = 0
            num_batches = number_of_names//self.hps.batch_size
            if num_batches == 0:
                num_batches = 1
                names_batch_size = number_of_names
            else:
                names_batch_size = self.hps.batch_size
            for _ in range(num_batches):
                names = self.generate_name(name_start,names_batch_size)
                for name in names:
                    if name in self.dataset.test.names:
                        score += 1
                    if print_names:
                        print(name)   
            score /= num_batches*names_batch_size/100
            print(f'score : {score:.3f}')
            return score

@dataclass
class HyperParams():
    device:str  = 'cuda'
    epochs: int = 50
    lr:float = 0.0001
    weight_decay:float = 0.0 #1e-5
    dropout_rate:float = 0.0
    lr_step:int = 1
    lr_decay:float = 1 # #0.5#0.85 # 0.85
    testing_num:int = 10000
    batch_size:int = 1
    emb_dim:int = 300
    hidden_size:int = 512
    num_layers:int = 2

def run(hps=HyperParams()):
    names   =   open('names.txt','r').read().splitlines()
    dataset =   Dataset(names,hps.device)    
    model   =   RNN(dataset,hps)
    trainer =   Trainer(hps,model,dataset)

    trainer.train(scoring = True ,saving = False)
    model.test_model(number_of_names=10000, print_names=False, name_start='')
    model.save('final')
    return

def gen():
    m = RNN.load('checkpoint')
    m.test_model(number_of_names=100,print_names=True,name_start='')
    return 

if __name__== '__main__':
    run()
    #gen()
    