#!/usr/bin/env python
# coding: utf-8

# In[1]:


# My First GAN

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# set up a random seed for neural networks weight initialization
torch.manual_seed(111)

# training data
n_train_data = 1024
train_data = torch.zeros((n_train_data, 2)) # (x, y = sin(x))
train_data[:, 0] = 2 * math.pi * torch.rand(n_train_data) # x = [0, 2*pi]
train_data[:, 1] = torch.sin(train_data[:, 0]) # y = sin(x)
train_labels = torch.zeros(n_train_data) # for Dataloader, not to be used later as GANs are unsupervised
train_set = [
    (train_data[i], train_labels[i]) for i in range(n_train_data)
] # for DataLoader


# examining dataset
plt.plot(train_data[:, 0], train_data[:, 1], '.') # sine curve
plt.show()

# Creating DataLoader, for convenient training later on
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)


# implementing discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),  # first layer input size -> (x, y) = 2
            nn.ReLU(),          # ReLu activation
            nn.Dropout(0.3),    # Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),       # probability of coming from original data
        )

    def forward(self, x):
        output = self.model(x)
        return output


discriminator = Discriminator()


# implementing generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16), # first layer input size = latent space = 2
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # output size -> (x, y) = 2
        )

    def forward(self, x):
        out = self.model(x)
        return out


generator = Generator()

# hyper parameters
learning_rate = 0.001
n_epochs = 300
# loss function
criterion = nn.BCELoss() # bcz of binary output of discriminator at the end of both training gen. or discriminator
# weights optimizer
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = learning_rate)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)

latent_space_fixed = torch.randn(100, 2)


# Training Loop
for epoch in range(n_epochs):
    for n, (real_samples, _ ) in enumerate(train_loader):
        # Create Data for Training the Discrim.
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator.forward(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples), dim = 0)
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels), dim=0)
        # Training the Discriminator
        discriminator.zero_grad() # empty out previous gradients
        output_discriminator = discriminator(all_samples) # get predicted labels
        loss_discriminator = criterion(output_discriminator, all_samples_labels) # calculate loss
        loss_discriminator.backward() # calc gradients
        optimizer_discriminator.step() # update the weights
        
        # Create Data for training Generator holding the discr. weights fixed
        latent_space_samples = torch.randn((batch_size, 2))
        # training the generator
        generator.zero_grad()
        generated_samples = generator.forward(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = criterion(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

    generated_fixed = generator(latent_space_fixed)
    generated_fixed = generated_fixed.detach()

    plt.plot(generated_fixed[:, 0], generated_fixed[:, 1], '.')
    plt.xlabel("x1")
    plt.ylabel("sin(x1)")
    plt.title(f"After {epoch} epoch(s)")
    plt.draw()
    plt.pause(0.1)
    plt.clf()


# generating new samples
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach() # no gradient calculation needed here
plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
plt.show()

