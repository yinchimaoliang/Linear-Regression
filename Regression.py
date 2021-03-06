import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = nn.Linear(1,100)
        self.predict = nn.Linear(100,1)
    def forward(self, x):
        out = F.relu(self.hidden(x)) #非常重要，去线性化的relu函数，否则永远是一条直线
        out = self.predict(out)
        return out

class Main():
    def getData(self):
        self.x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)#增加一维，变为二维
        # self.x = torch.linspace(-1,1,100)
        self.y = self.x.pow(3) + 1 +  0.3 * torch.rand(self.x.size())#添加噪点
        # plt.scatter(self.x.data.numpy(),self.y.data.numpy())
        # plt.show()
    def main(self):
        model = Net().cuda()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.2)
        loss_func = nn.MSELoss()
        plt.ion()
        for epoch in range(EPOCH):
            inputs = self.x.cuda()
            targets = self.y.cuda()

            out = model(inputs)
            loss = loss_func(out,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                plt.cla()
                predict = model(self.x.cuda())
                plt.scatter(self.x.data.numpy(),self.y.data.numpy())
                plt.plot(self.x.data.numpy(),predict.data.cpu().numpy()) #必须转换为cpu才能转换为numpy
                plt.pause(1)

        plt.ioff()
        plt.show()






t = Main()
t.getData()
t.main()
