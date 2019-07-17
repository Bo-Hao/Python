import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F



# fake data
n_data = torch.ones(100, 2)         # data common form
x0 = torch.normal(2*n_data, 1)      # type 0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # type 0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # type 1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # type 1 y data (tensor), shape=(100, 1)

# we should set datas as the form below  (torch.cat is merging)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# torch can only train with Variable. Thus, turn they into Variable
x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):     # inherit torch's Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # inherit __init__ 
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)       # output layer

    def forward(self, x):
        # forward transpose
        x = F.relu(self.hidden(x))      # activation function
        x = self.out(x)                 # output, this is not prediction.
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) # type number = classification number output

print(net)  # net struction



# optimizer is the trainning tool.
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # lr = learning rate
# 算誤差的时候, 注意真實值!不是!, is 1-D Tensor, (batch,)
# prediction is 2-dimension. tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()



plt.ion()   # graph
plt.show()

for t in range(100):
    out = net(x)     # training  x, output

    loss = loss_func(out, y)     # calculating loss.

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

    if t % 2 == 0:
        plt.cla()
        # prediction will be after softmax (activation functuon)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200  # comparing
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # stop graphing
plt.show()

print(net(x))