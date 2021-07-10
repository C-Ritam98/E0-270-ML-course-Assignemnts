''' 
 Vanilla GAN on Fashion MNIST dataset
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

from torchvision import datasets,transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    """
        Simple Discriminator :
        NN architechture: [ip-layer hidden1 hidden2 op-layer] = [input-size 512 256 1]
    """
    def __init__(self, input_size=784, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        y_ = x.view(x.size(0), -1)
        y_ = self.layer(y_)
        return y_


class Generator(nn.Module):
    """
        Simple Generator :
        NN architechture: [ip-layer hidden1 hidden2 hidden3 hidden4 op-layer] = [input-size 128 256 512 1024 784]
    """
    def __init__(self, input_size=100, num_classes=784):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),
            nn.Tanh()
        )
        
    def forward(self, x):
        y_ = self.layer(x)
        y_ = y_.view(x.size(0), 1, 28, 28)
        return y_


def Visualise_img(G, n_noise):
    """
        save sample 100 images
    """
    z = torch.randn(100, n_noise).to(DEVICE)
    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([280, 280])
    for j in range(10):
        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=1)
    return img




n_noise = 100

Disc_obj = Discriminator().to(DEVICE)
Gen_obj = Generator(n_noise).to(DEVICE)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(.5,))])

fmnist_train = datasets.FashionMNIST(root = "./data/",train = True, transform = transform, download = True)
fmnist_test = datasets.FashionMNIST(root = "./data/", train = False, download = True)

batch_size = 120

train_loader = DataLoader(dataset = fmnist_train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = fmnist_test, batch_size = batch_size, shuffle = True)

Disc_optim = torch.optim.Adam(Disc_obj.parameters(), lr = 3e-4)
Gen_optim = torch.optim.Adam(Gen_obj.parameters(), lr = 3e-4)

update_Gen_after = 1
count = 0
epochs = 50

Disc_correct = torch.ones(batch_size,1).to(DEVICE)
Disc_fake = torch.zeros(batch_size,1).to(DEVICE)

if not os.path.exists('Outputs'):
    os.makedirs('Outputs')

criterion = nn.BCELoss()


# train

D_loss = []
G_loss = []
for epoch in range(epochs):
    for _,(images,labels) in enumerate(train_loader):

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # train discriminator

        Disc_output = Disc_obj(images)

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

        if count % 500 == 0:
            print(f"epoch:{epoch}/{epochs}, D_loss:{total_loss.item()}, G_loss:{gen_loss.item()}")
            D_loss.append(total_loss.item())
            G_loss.append(gen_loss.item())
            Gen_obj.eval()
            img = Visualise_img(Gen_obj, n_noise)
            plt.imsave(f'Outputs/epoch{epoch}_step{count}.jpg', img, cmap='gray')
            Gen_obj.train()


        count += 1


plt.plot(D_loss,label = "Discriminator Loss")
plt.plot(G_loss, label = "Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.title("Loss vs Epoch plot")
plt.show()
