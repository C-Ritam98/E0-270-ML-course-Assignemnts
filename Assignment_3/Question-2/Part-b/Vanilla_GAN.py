''' 
 Vanilla GAN on point dataset
'''
#from matplotlib.colors import LightSource, LinearSegmentedColormap
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

from torchvision import datasets,transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Custom_Dataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        with open("gan_data.txt",'r') as f:
            lines = f.readlines()
            data = np.array([[float(y) for y in x.split(',')] for x in lines]).reshape(-1,2)

        plt.scatter(data[:,0],data[:,1])
        plt.show()
        
        self.n_samples = data.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(data[:, 0:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(np.ones((self.n_samples,1))) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



class Discriminator(nn.Module):
    """
        Simple Discriminator :
        NN architechture: [ip-layer hidden1 hidden2 op-layer] = [input-size 512 256 1]
    """
    def __init__(self, input_size=2, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
             nn.Linear(input_size,  32),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(32, 512),
             nn.ReLU(),            
             nn.Dropout(0.4),
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(256, 128),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(64, num_classes),
             nn.Sigmoid(),
        )
    
    def forward(self, x):
        y_ = x.view(x.size(0), -1)
        y_ = self.layer(y_.float())
        return y_


class Generator(nn.Module):
    """
        Simple Generator :
        NN architechture: [ip-layer hidden1 hidden2 hidden3 hidden4 op-layer] = [input-size 128 256 512 1024 784]
    """
    def __init__(self, input_size=10, num_classes=2):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 8 ),

            nn.ReLU(),
            nn.Linear(8, 16),
            
            nn.ReLU(),
            nn.Linear(16, 8),
            
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )
        
    def forward(self, x):
        y_ = self.layer(x.float())
        y_ = y_.view(x.size(0), 2)
        return y_


def Visualise_img(G, n_noise):
    """
        save sample 100 images
    """
    z = torch.randn(500, n_noise).to(DEVICE)
    y_hat = G(z).view(500, 2)
    result = y_hat.cpu().data.numpy()
    
    x = []
    y = []
    for i in range(500):
        x.append(result[i,0])
        y.append(result[i,1])

    plt.scatter(x,y)
    plt.title("GAN generated")
    plt.savefig(f'Outputs_a/epoch{epoch}_step{count}.jpg')
    plt.clf()




n_noise = 2

Disc_obj = Discriminator().to(DEVICE)
Gen_obj = Generator(n_noise).to(DEVICE)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(.5,))])

dataset = Custom_Dataset()

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)



batch_size = 50

data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

Disc_optim = torch.optim.Adam(Disc_obj.parameters(), lr = 1e-4)
Gen_optim = torch.optim.Adam(Gen_obj.parameters(), lr = 1e-4)

update_Gen_after = 1
count = 0
epochs = 1000

Disc_correct = torch.ones(batch_size,1).to(DEVICE)
Disc_fake = torch.zeros(batch_size,1).to(DEVICE)

if not os.path.exists('Outputs_a'):
    os.makedirs('Outputs_a')

criterion = nn.BCELoss()


# train

D_loss = []
G_loss = []
for epoch in range(epochs):
    for _,(points,labels) in enumerate(data_loader):

        points, labels = points.to(DEVICE), labels.to(DEVICE)
        
        # train discriminator

        Disc_output = Disc_obj(points)

        correct_loss = criterion(Disc_output, Disc_correct)
        
        z = torch.randn(batch_size, n_noise).to(DEVICE)
        Gen_output = Disc_obj(Gen_obj(z))
        fake_loss = criterion(Gen_output, Disc_fake)

        total_loss = fake_loss + correct_loss

        Disc_optim.zero_grad()
        total_loss.backward()
        Disc_optim.step()

        # update Generator after 'update_Gen_after' updations in Discriminant
        if count % update_Gen_after == 0 :

            z = torch.randn(batch_size,n_noise).to(DEVICE)

            Gen_output = Disc_obj(Gen_obj(z))
            gen_loss = criterion(Gen_output, Disc_correct)

            Gen_optim.zero_grad()
            gen_loss.backward()
            Gen_optim.step()

        if count % 10 == 0:
            print(f"epoch:{epoch}/{epochs}, D_loss:{total_loss.item()}, G_loss:{gen_loss.item()}")
            D_loss.append(total_loss.item())
            G_loss.append(gen_loss.item())
            Gen_obj.eval()
            Visualise_img(Gen_obj, n_noise)
            #plt.imsave(f'Outputs_a/epoch{epoch}_step{count}.jpg', img, cmap='gray')
            Gen_obj.train()


        count += 1


plt.plot(D_loss,label = "Discriminator Loss")
plt.plot(G_loss, label = "Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.title("Loss vs Epoch plot")
plt.show()
