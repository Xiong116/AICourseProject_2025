import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24, kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3,padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.branch5x5_2 = nn.Conv2d(16,24, kernel_size=5, padding=2)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self,x):
        #分支1
        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)

        #分支2
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        #分支3
        branch1x1=self.branch1x1(x)

        #分支4
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        #将4个分支的输出拼接起来（输出为88通道）
        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(outputs,dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)
        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(2200,10)  #5*5*88=2200，图像高*宽*通道数
        
    def forward(self,x):
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x = x.view(-1, 2200)
        x=self.fc(x)

        return x

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


batch_size = 64
batch = 300
epochs = 50
#数据提取
trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
##数据打包
trainloader=DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True
)

testset=datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader=DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False)

model=Net()
model=model.cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    for batch_idx,data in enumerate(trainloader,0):
        images,labels=data
        images=images.cuda()
        labels=labels.cuda()

        outputs=model(images)
        optimizer.zero_grad()
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if batch_idx % batch == 0:
                print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                    epoch,100. * batch_idx / len(trainloader), loss.item()
                ))
        
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in  testloader:
            images,labels=data
            images=images.cuda()
            labels=labels.cuda()

            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    
    print('Accuracy on test set:%d %%'%(100*correct/total))

if __name__ == '__main__':
    for epoch in range(1,epochs+1):
        train(epoch)
        test()