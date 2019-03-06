import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

POLY_DEGREE = 3
EPOCH = 10000

class PolyRegression():
    def get_features(self,x):
        self.x = torch.unsqueeze(x,1)
        return torch.cat([self.x ** i for i in range(1,POLY_DEGREE + 1)],1)#将矩阵按列拼接起来
    def get_batch(self,size):
        random = torch.linspace(-1,1,size)
        self.mat = self.get_features(random)
        self.x = self.x.cuda()
        self.mat = self.mat.cuda()
        # plt.scatter(self.x.data.numpy(),self.y.data.numpy())
        # plt.show()
    def get_data(self):
        w = torch.FloatTensor([3,2,1]).unsqueeze(1).cuda()
        b = torch.FloatTensor([1]).cuda()
        self.y = self.mat.mm(w) + b[0] + 0.5 * torch.rand(self.x.size()).cuda()
    def main(self):
        self.get_batch(100)
        self.get_data()
        net = PolyNet().cuda()
        optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
        loss_func = nn.MSELoss()
        plt.ion()
        for epoch in range(EPOCH):
            predicts = net(self.mat)
            loss = loss_func(predicts,self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                plt.cla()
                plt.scatter(self.x.data.cpu().numpy(),self.y.data.cpu().numpy())
                plt.plot(self.x.data.cpu().numpy(),predicts.data.cpu().numpy())
                plt.pause(0.2)

        for i in range(POLY_DEGREE):
            print(net.predict.weight.view(-1)[i].data.cpu().numpy(),end = " ")#打印出参数
        print(net.predict.bias[0].data.cpu().numpy())
        plt.ioff()
        plt.show()
class PolyNet(nn.Module):
    def __init__(self):
        super(PolyNet,self).__init__()
        self.predict = nn.Linear(3,1)
    def forward(self, x):
        out = self.predict(x)
        return out



if __name__ == "__main__":
    t = PolyRegression()
    t.main()