from .BasicModule import BasicModule
import torch.nn as nn



class Perceptron(BasicModule):

    def __init__(self,in_dim,hidden_dim,out_dim):
        super(Perceptron,self).__init__()

        self.model_name = 'perceptron'

        self.lin1 = nn.Linear(in_dim,hidden_dim)
        self.sig = nn.Sigmoid()
        self.lin2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.sig(x)
        x = self.lin2(x)
        return x