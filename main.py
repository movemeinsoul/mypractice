import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      
from model import CNN
#import numpy as np
 
torch.manual_seed(1)    # reproducible
 
# Hyper Parameters
EPOCH = 5          # 训练整批数据多少次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就fasle

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    
    download=DOWNLOAD_MNIST       # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
 
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
 
# 为了节约时间, 我们测试时只测试前1000个
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:1000]/255.   # shape from (1000, 28, 28) to (1000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:1000]

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)       # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)



#test_output = cnn(test_x)
#pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
#print((np.asarray(pred_y)==np.asarray(test_y)).sum()/1000)

#print(pred_y, 'prediction number')
#print(test_y[:10].numpy(), 'real number')
