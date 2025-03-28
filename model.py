import torch 
from torch import nn # Import neural network module

class NetWork(nn.Module):
    def __init__(self):
        super().__init__()  # Call the constructor of the parent class
        self.layer1 = nn.Linear(784, 256)   #Define first linear layers ( input size 784, output size 256)
        self.layer2 = nn.Linear(256, 10)    #Define second linear layers ( input size 256, output size 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)     # Flatten the input tensor to 1D tensor of size 28*28 (equals to 784)
        x = self.layer1(x)  #Transform the input tensor using the first linear layer
        x = torch.relu(x)   #Convert the input tensor to non-linear space using ReLU activation function
        return self.layer2(x)   # Pass the sesult through the second linear layer and return the output tensor