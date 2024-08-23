import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.sgd

import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import time
from torch.utils.tensorboard import SummaryWriter




if __name__ == '__main__':
    training = 0.8
    dev = 0.1
    ## LOAD NAMES ##
    names = open('names.txt','r').read().splitlines()
    chars = sorted(list(set(''.join(names))))
    chars.insert(0,'.')
    int_to_char = {i:chars[i] for i in range(len(chars))}
    char_to_int = {c:i for i,c in int_to_char.items()}
    num_chars = len(int_to_char)

    random.seed(1)
    random.shuffle(names)
    n1 = int(len(names)*training)
    n2 = int(len(names)*dev) + n1
    training_names = names[:n1]
    dev_names = names[n1:n2]
    test_names = names[n2:]
    writer = SummaryWriter()

def create_dataset(names,block_size,char_to_int):
    X = []
    Y = []
    for name in names:
        context = [0]*block_size
        for char in name + '.':
            index = char_to_int[char]
            X.append(context)
            Y.append(index)
            context = context[1:] + [index]
    return  torch.tensor(X), torch.tensor(Y)

def moving_average(array, n):
    if n % 2 == 0:
        raise  ValueError('n must be odd' )
    return np.convolve(np.array(array),np.ones(n))[n-1:-(n-1)]/n

class NeuralNetwork(nn.Module):
        def __init__(self,num_chars,embedding_size,block_size,num_neurons):
            self.block_size = block_size
            super().__init__()
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            self.sequential_stack = nn.Sequential(
                nn.Embedding(num_chars,embedding_size),
                nn.Flatten(1),
                #nn.Dropout(p=0.2),
                nn.Linear(block_size*embedding_size,num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons,num_chars)
            )
            super().to(self.device)
            self.loglosses = []

        def forward(self,x):
            if x.device != self.device:
                x = x.to(self.device)

            x = self.sequential_stack(x)
            return x
        
        def loss(self,x,y):
            if x.device != self.device:
                x = x.to(self.device)
                y = y.to(self.device)

            logits = self.forward(x)
            loss = F.cross_entropy(logits,y)
            return loss
        
        def train(self,epochs,batch_size,x,y,learning_rate,x_dev,y_dev):
            start = time.time()
    
            if x.device != self.device:
                x = x.to(self.device)
                y = y.to(self.device)
                
            num_its = (len(x) * epochs) //batch_size
            

            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

            for it in range(num_its):
                print(f'training... {it:06d}/{num_its:06d}',end='\r')
                batch_indices = torch.randint(0,x.shape[0],(batch_size,))
                x_batch = x[batch_indices]
                y_batch = y[batch_indices]

                self.forward(x_batch)

                loss = self.loss(x_batch,y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loglosses.append(loss.log10().item())
                writer.add_scalar("logloss/train",loss,it)   
                if (it*100)//num_its %10 == 0:
                    writer.add_scalar('logloss/dev',self.loss(x_dev,y_dev),it/num_its)            
            end = time.time()
            self.time = end - start
            
            return
        
        def generate_name(self,name_start=''):
            name = name_start
            if len(name) < self.block_size:
                name = '.' * ( self.block_size - len(name)) + name
            context = [c for c in name][:self.block_size]
            #context = ['.'] * block_size

            context = [char_to_int[i] for i in context]
            out = []
            while True:
                logits = self.forward(torch.tensor([context]).to(device=self.device))
                probs = F.softmax(logits,dim=1)
                ix = torch.multinomial(probs,num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    out.pop(-1)
                    break

            output = name_start+''.join(int_to_char[i] for i in out)
            return output

        def test_model(self,test_names=None,number_of_names = 100000,name_start = '',print_names = False):

            score = 0

            for i in range(number_of_names):
                #print(f'testing... {i:06d}/{number_of_names:06d}',end='\r')
                name = self.generate_name(name_start)
                if print_names:
                    print(name)
                if name in test_names:
                    score += 1
                    #print(name)
            score /= number_of_names/100
            return score

def run(
    epochs = 200,
    learning_rate = 0.0001,
    block_size = 5,
    embedding_size = 300,
    num_neurons = 512,
    batch_size = 2048
    ):

    

    X_train, Y_train = create_dataset(training_names,block_size,char_to_int)
    X_dev, Y_dev = create_dataset(dev_names,block_size,char_to_int)
    X_test, Y_test = create_dataset(test_names,block_size,char_to_int)
    ####

    m = NeuralNetwork(num_chars,embedding_size,block_size,num_neurons)
    m.train(epochs,batch_size,X_train,Y_train,learning_rate,X_dev,Y_dev)
    loss = m.loss(X_dev,Y_dev)

    time_elapsed = m.time
    
    score = m.test_model(   test_names=test_names,
                            number_of_names=10,
                            print_names=True,
                            name_start='')

    vars = [m, X_test, Y_test,int_to_char,char_to_int,block_size,training_names, test_names,score]
    
    
    with open('objs.pkl', 'wb') as f: 
        pickle.dump(vars, f)
    

    print(  f'\nTIME: {time_elapsed:.3f}s\n'
            f'LOSS: {loss.item():.3f}\n'
            f'SCORE: {score:.2f}%\n')
    writer.flush()
    return loss.item(), time_elapsed,score

def gen():
    with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
        m, X_test, Y_test,int_to_char,char_to_int,block_size,training_names,test_names = pickle.load(f)
    score = m.test_model(test_names=test_names,number_of_names=50,
                        print_names=True,name_start='')
    print(score)
    return score

if __name__== '__main__':
    run()
    #gen()
    
    # xs = [10,50,100,150,200,300,400,500,1000]
    # ls, ss = [], []
    # for n in xs:
    #     l, t, s = run(epochs=n)
    #     ls.append(l)
    #     ss.append(s)
    # plt.plot(xs,ls/np.mean(ls))
    # plt.plot(xs,ss/np.mean(ss))
    # plt.show()

    