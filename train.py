import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import Mynet
import torch

# tensorboard
writer=SummaryWriter("./logs_train")

# 用cuda训练，模型，数据，损失函数放入cuda中
device = torch.device("cuda")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True,transform=torchvision.transforms.ToTensor(),download=False)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False,transform=torchvision.transforms.ToTensor(),download=False)
# length
train_data_size=len(train_data)
test_data_size=len(test_data)
# print(train_data_size, test_data_size)

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络
mynet = Mynet()
mynet = mynet.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(mynet.parameters(),lr=learning_rate)

train_step=0
test_step=0
# 开始训练
epoch=10
for i in range(epoch):
    mynet.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs=imgs.to(device)
        targets=targets.to(device)
        output = mynet(imgs)
        loss = loss_function(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step+=1
        print(f"第{train_step}轮: loss{loss}")
        writer.add_scalar("train_loss",loss.item(),train_step)

    # 对模型进行测试
    mynet.eval()
    test_loss=0
    accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs=imgs.to(device)
            targets=targets.to(device)
            output=mynet(imgs)
            loss=loss_function(output, targets)
            test_loss=test_loss+loss.item()
            accuracy=accuracy+(output.argmax(1)==targets).sum()
    print(f'loss{test_loss} accuracy{accuracy/test_data_size}')
    writer.add_scalar("test_loss",test_loss,test_step)
    test_step+=1
    torch.save(mynet,"pth/mynet_{}.pth".format(i))
writer.close()