torch.cuda.is_available()


# In[ ]:


conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


# In[2]:


import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision



# In[4]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[14]:


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 7)
        self.pool1 = nn.MaxPool2d(2,stride=1)
        self.conv2 = nn.Conv2d(10, 21, 5)
        self.pool2=nn.MaxPool2d(4,stride=2)
        self.fc= nn.Linear(1701, 10)
        
        
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x


# In[15]:


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epochs=2
loss_list=[]
tot_loss=0


# In[16]:


#n_epochs
for epoch in range(n_epochs):
      for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #clear gradient 
            optimizer.zero_grad()
            #make a prediction 
            z=net(inputs)
            loss=criterion(z,labels)
            # calculate gradients of parameters 
            loss.backward()
            # update parameters 
            optimizer.step()

            loss_list.append(loss.data)
            tot_loss+=loss.data
            if(i%2000==1999):
                  print(f'[{epoch+1}, {i+1:5d}] loss: {tot_loss/len(loss_list)}')


# In[17]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy is: {100 * correct // total} %')
