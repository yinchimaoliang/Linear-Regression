import torch
import matplotlib.pyplot as plt
import numpy as np




class Main():
    def getData(self):
        self.x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
        # self.x = torch.linspace(-1,1,100)
        self.y = self.x * 3 + 1 +  torch.rand(self.x.size())
        plt.scatter(self.x.data.numpy(),self.y.data.numpy())
        plt.show()





t = Main()
t.getData()
