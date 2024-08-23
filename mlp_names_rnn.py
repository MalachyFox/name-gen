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

class NamesToData():
            def __init__(self,dataset,names):
                X, Y = [], []
                for name in names:
                    context = [0]*dataset.block_length
                    for char in name + '.':
                        index = dataset.char_to_int[char]
                        X.append(context)
                        Y.append(index)
                        context = context[1:] + [index]
                self.x = torch.tensor(X).to(dataset.device)
                self.y = torch.tensor(Y).to(dataset.device)
                self.names = names

class Dataset():
    def __init__(self,names,block_length,device,train_split = 0.8,dev_split = 0.1):
        self.block_length = block_length
        self.device = device
        chars = sorted(list(set(''.join(names))))
        chars.insert(0,'.')
        self.int_to_char = {i:chars[i] for i in range(len(chars))}
        self.char_to_int = {c:i for i,c in self.int_to_char.items()}
        self.num_chars = len(self.int_to_char)

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
                score = self.test_model(self.hps.testing_batch_size)
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
        print(f'time elapsed: {time_elapsed:.1f}')
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
            x = x[:,-1,:]
            x = self.linear(x)
            return x
        
        def loss(self,*args):
            if isinstance(args[0],NamesToData):
                data = args[0]
                x, y = data.x, data.y
            else:
                x,y = args    

            logits = self.forward(x)
            loss = F.cross_entropy(logits,y)
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
            if len(name) < self.hps.block_length:
                name = '.' * ( self.hps.block_length - len(name)) + name
            context = [c for c in name][:self.hps.block_length]
            context = [[self.dataset.char_to_int[i] for i in context]]

            context = torch.tensor(context,device=self.hps.device)
            context = context.expand(batch_size,-1)
            max_name_length = 20
            out = torch.empty(batch_size,max_name_length)
            for char_i in range(max_name_length):
                logits = self.forward(context)
                probs = F.softmax(logits,dim=1)
                new_char = torch.multinomial(probs,num_samples=1).to(device=self.hps.device)
                context = torch.cat((context[:,1:],new_char),dim=1)
                out[:,char_i] = new_char.squeeze()
            output = []
            for row in out:
                name = name_start+''.join(self.dataset.int_to_char[i.item()] for i in row)
                name = name.split('.')[0]
                output.append(name)
            return output

        def test_model(self,number_of_names = 1e6,name_start = '',print_names = False):
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
            return

@dataclass
class HyperParams():
    device:str  = 'cuda'
    epochs: int = 15
    lr:float = 0.001
    weight_decay:float = 1e-5
    dropout_rate:float = 0.0
    lr_step:int = 1
    lr_decay:float = 0.85
    testing_batch_size:int = 10000
    batch_size:int = 512

    block_length:int = 8
    emb_dim:int = 300
    hidden_size:int = 512
    num_layers:int = 2

def run(hps=HyperParams()):
    names   =   open('names.txt','r').read().splitlines()
    dataset =   Dataset(names,hps.block_length,hps.device)    
    model   =   RNN(dataset,hps)
    trainer =   Trainer(hps,model,dataset)

    trainer.train(scoring = False,saving = True)
    model.test_model(number_of_names=100000, print_names=False, name_start='')
    model.save('final')
    return

def gen():
    m = RNN.load('checkpoint')
    m.test_model(number_of_names=100,print_names=True,name_start='mal')
    return 

if __name__== '__main__':
    #run()
    gen()
    