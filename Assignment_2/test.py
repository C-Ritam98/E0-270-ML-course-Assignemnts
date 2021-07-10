# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import pandas as pd
import numpy as np
from functools import reduce
from train import ConvNet,index_to_name

import seaborn as sns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # load model
    model = torch.load("./Model")
    model.eval()

    # convert to tensor and normalise the data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    mnist_test = torchvision.datasets.FashionMNIST(root='./data/',train=False,download = True,transform = transform)
    test_loader = torch.utils.data.DataLoader(mnist_test,batch_size = 64)

    n_correct = 0
    n_samples = 0
    n_class_predict = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    n_class_correct = [0 for i in range(10)]
    predicted = []
    actual = []

    for i,(images,labels) in enumerate(test_loader):  # predict in batches of 64
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        prediction = torch.max(outputs,1)[1]
        
        n_samples += len(labels) #.size(0)
        n_correct += (prediction == labels).sum()

        for i in range(len(labels)):
            label = labels[i]
            pred = prediction[i]
            n_class_samples[label]+=1
            n_class_correct[label] += int(pred == label)
            n_class_predict[pred] += 1
            predicted.append(pred)
            actual.append(label)

    print(f'Accuracy = {n_correct/n_samples}')
    print("Predicted:\t",n_class_predict)
    print("Correctly sampled in classes",n_class_correct)
    print("Actual distribution of classes:",n_class_samples)

    for i in range(10):
        print(index_to_name(i), end = ' ')


    textfile = open("cnn.txt", "w")
    for element in predicted:
        textfile.write(str(element.numpy()) + '\n')
    textfile.close()

    print(' ')
    print(confusion_matrix(actual,predicted))

    sns.heatmap(confusion_matrix(actual,predicted),annot = True)
    plt.show()
