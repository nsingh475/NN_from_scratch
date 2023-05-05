#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import defaultdict
training_set = [
    [[0, 0,], [0,  0, ]],
    [[1, 1,], [1,  1, ]],
    [[0,], [0, ]],
    [[1,], [1, ]],
    [[0, 0,], [0,  0, ]],
    [[1, 1,], [1,  1, ]],
    [[0,], [0, ]],
    [[1,], [1, ]],
    [[0, 0,], [0,  0, ]],
    [[1, 1,], [1,  1, ]],
    [[0,], [0, ]],
    [[1,], [1, ]],
    [[0, 0,], [0,  0, ]],
    [[1, 1,], [1,  1, ]],
    [[2,], [2, ]],
    [[1,], [1, ]],
]

def oh(n):
    x = [0,0,0]
    x[n] = 1
    return x
training_set = [[torch.Tensor([oh(n) for n in l2[0]]), torch.Tensor(l2[1]).type(torch.LongTensor)] for l2 in training_set]

class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 4)
        self.rnn = nn.RNN(4, 3, batch_first=True)
        self.output = nn.Linear(3, 3)
        self.soft = nn.Softmax(dim=0)
    def forward(self, inp):
        i2 = self.embed(inp).unsqueeze(0)
        l_output, _ = self.rnn(i2)
        o = self.output(l_output).squeeze(0)
        s = self.soft(o)
        return s

class Trainer():
  def __init__(self, lr=0.01):
      self.rnn = MyRNN()
      # use gradient clipping for some reason
      torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1)
      self.optimizer = torch.optim.Adam(self.rnn.parameters(), amsgrad=True, lr=lr)
      self.losses = []
      self.gradients = []
  def epoch(self):
      x = len(self.losses)
      for i in range(len(training_set)):
        input, target = training_set[i]
        # forward pass
        output = self.rnn(input)
        loss = F.cross_entropy(output[-1:], target[-1:])
        # compute gradients and take optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # save the losses & gradients so we can graph them if we want
        self.losses.append(loss.item())
        self.gradients.append(torch.norm(self.rnn.rnn.weight_hh_l0.grad))
      # Print the loss for this epoch
      print(np.array(self.losses[x:]).mean())
      # Print out some predictions
      #print(''.join(make_preds(self.rnn, temperature=1)))

def pred(model, inp):
    return torch.multinomial(model(inp), 1).flatten()

def acc(model):
    total = 0
    correct = 0
    for inp, exp in training_set:
        out = pred(model, inp)
        for a, b in zip(exp, out):
            total += 1
            if a == b:
                correct += 1
    print(f'{correct} / {total} = ~{round(100*correct/total,2)}%')

trainer = Trainer()
for i in range(2000):
    print(f'epoch {i+1}')
    trainer.epoch()
    acc(trainer.rnn)
