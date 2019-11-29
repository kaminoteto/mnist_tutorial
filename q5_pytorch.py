import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import time

#                          .:::::::::::::.
#                        .:::::::::::::::::.
#                      .:::::::::::::::::::::.
#                     :::::::::::::::::::::::::
#                  ..::::::( ◕ )::::( ◕ ):::::::..
#                 :::＠:::::::::(●●)::::::::::::＠:::
#                 .::::::::::::::__:::::::::::::::::.
#                ..:::::::::::::::::::::::::::::::::..
#                ::::::::      ::::::::      :::::::::
#                ::::::::      ::::::::      :::::::::
#                ::::::::      ::::::::      :::::::::
#                      ::::::::::::::::::::::
#                ::::``::::::::::::::::::::::``::::
#               ::::'  ::::::::::::::::::::::  '::::
#             .::::'   ::::::::::::::::::::::   '::::.
#            .:::'      ::::::::::::::::::::      ':::.
#           .::'        ::::::::::::::::::::        '::.
#          .::'         ::::::::::::::::::::         '::.
#      ...:::           ::::::::::::::::::::           :::...
#     ```` ':.          '::::::::::::::::::'          .:' ````
#                        '.::::::::::::::.'
#                          榕榕保佑 永无BUG

BATCH_SIZE = 128
NUM_EPOCHS = 10
learning_rate = 0.02

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)



# TODO:define model
class SimpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#model = SimpleNet()


model = SimpleNet(28 * 28, 300, 100, 10)

if torch.cuda.is_available():
    model = model.cuda()

# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train the model
epoch = 0
for data in train_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))

