
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import helper

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super(Network, self).__init__()
        
        # Input to first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add arbitrary number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Last hidden layer to output
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        # Flatten input tensor
        x = x.view(x.shape[0], -1)
        
        # Pass through each hidden layer with ReLU and dropout
        for linear in self.hidden_layers:
            x = self.dropout(F.relu(linear(x)))
        
        # Output layer with log softmax activation
        x = F.log_softmax(self.output(x), dim=1)
        
        return x



def train(model, trainloader, testloader, criterion, optimizer, epochs=5):
    steps = 0
    train_losses, test_losses = [], []
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            model.train()  # Set model to training mode
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            # Validation pass
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()  # Set model to evaluation mode
                for images, labels in testloader:
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels).item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
    return train_losses, test_losses


