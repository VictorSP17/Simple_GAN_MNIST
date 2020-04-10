# # Generative Adversarial Networks - GAN's
#
# In this notebook we will generate images of handwritten digits using the MNIST dataset.

# ## Table of Contents
#
# 1. [Import libraries and define auxiliar functions](#1st)
# 2. [Hyperparameters of the model (easy to modify)](#2nd)
# 3. [Data preprocessing and EDA](#3rd)
# 4. [Building the model](#4th)
# 5. [Hyperparameters of the model (easy to modify)](#5th)
# 6. [Hyperparameters of the model (easy to modify)](#6th)

# <div id='1st'/> 
#
# ## 1. Import libraries and define auxiliar functions

# Let us first import the necessary libraries:

import os
import torch #
import torchvision #
import torch.nn as nn
from torchvision import transforms #
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
import sys
from inception_score import inception_score
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Now, the functions auxiliar functions we will need:

# +
# This
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# -

# <div id='2nd'/> 
#
# ## 2. Hyperparameters of the model (easy to modify)

# These are the hyperparameters of the model:

latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 1 # Initially 300
batch_size = 32
sample_dir = 'samples'
save_dir = 'save'
interpolate = False
NN = False

# <div id='3rd'/>
#
# ## 3. Data preprocessing and EDA: exploring our dataset

# Here we clean the data and process it so that we can work with it:

# The original pixel intensities range from 0 to 1. Using this transformation it switches to -1, 1.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std = (0.5,))])
mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# 'mnist' is the training set of the MNIST dataset:

print(mnist)

# Recall that MNIST images have dimension 28x28

print(mnist[0][0][0][0])

# Here we can see the first data point. This is a tuple (image, target). In this case the number is 5.

print(mnist[0]) 

# This is the tensor image

print(mnist[0][0].shape)

image = mnist[0][0].reshape(28,28)
print(image.shape)

plt.imshow(image, cmap="gray")

print("The pixel intensities range from", int(image.min()), "to", int(image.max()))

# Now that we know the structure of our dataset we can start building the model.

# <div id='4th'/>
#
# ## 4. Building the model

# Now, let us define the networks we will work with:

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# If we wanted to run this with a GPU we could then do:
#
# ``` python
# D = D.cuda()
# G = G.cuda()
# ```

# Let us now set the criteria and the optimizer:

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# Now we initialize statistics that we want to keep track of:

# Statistics to be saved
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

total_step = len(data_loader)

# If we have models saved we can load them:

# Load the model
state_dict = torch.load('save/G--100.ckpt')
G.load_state_dict(state_dict)
state_dict = torch.load('save/D--100.ckpt')
D.load_state_dict(state_dict)

# Let us train our model now :)

# TRAINING!
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.view(batch_size, -1) #cuda()
        images = Variable(images)
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1) #.cuda()
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(batch_size, 1) #.cuda()
        fake_labels = Variable(fake_labels)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size) #.cuda()
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        # If D is trained so well, then don't update
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size) #.cuda()
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        # if G is trained so well, then don't update
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # =================================================================== #
        #                          Update Statistics                          #
        # =================================================================== #
        d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.data*(1./(i+1.))
        g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.data*(1./(i+1.))
        real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().data*(1./(i+1.))
        fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().data*(1./(i+1.))

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.data, g_loss.data,
                          real_score.mean().data, fake_score.mean().data))

    # Save real images
    if (epoch+1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image(denorm(images.data), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images, this are the last ones instanced in the training...
    fake_images = denorm(fake_images.view(fake_images.size(0), 1, 28, 28)).round() # rounded
    save_image(fake_images.data, os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    # Save and plot Statistics
    np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
    np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
    np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
    np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
    plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.pdf'))
    plt.close()

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
    plt.plot(range(1, num_epochs + 1), real_scores, label='real score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
    plt.close()

    # Save model at checkpoints
    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(), os.path.join(save_dir, 'G--{}.ckpt'.format(epoch+1)))
        torch.save(D.state_dict(), os.path.join(save_dir, 'D--{}.ckpt'.format(epoch+1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

# Extra things...

# Once the model has been trained, let us interpolate two random points in the latent space.
# z_1*alpha + (1-alpha)*z_2, alpha from 0 to 1
if interpolate == True:
	z1 = torch.randn(latent_size) #.cuda()
	z1 = Variable(z1)
	z2 = torch.randn(latent_size) #.cuda()
	z2 = Variable(z2)
	for i in range(0,51): # alpha = i/50
		alpha = i/50.0
		z = (1-alpha)*z1 + alpha*z2
		fake_image = G(z)
		fake_image = denorm(fake_image.view(1, 28, 28)).round() # rounded
		save_image(fake_image.data, os.path.join(sample_dir, 'interp-{}.png'.format(i)))

# NEAREST NEIGHBOURS:
# Images 28x28, search the closest one.
# function(generated_image) --> closest training_image
if NN == True:
	aux_data_loader = torch.utils.data.DataLoader(dataset=mnist,
	                                              batch_size=1,
	                                              shuffle=False)

	def nearest_gt(generated_image):
	    min_d = 0
	    closest = False
	    for i, (image, _) in enumerate(aux_data_loader):
	        image = denorm(image.view(1, 28, 28)).round() # all distances in binary...
	        d = torch.dist(generated_image,image) # must be torch tensors (1,28,28)
	        if i == 0 or d < min_d:
	            min_d = d
	            closest = image

	    return closest

	# Calculate NN musaic 8x3
	z = torch.randn(24, latent_size)
	fake_images = G(z)
	fake_images = denorm(fake_images.view(24, 1, 28, 28)).round()
	save_image(fake_images.data, os.path.join(sample_dir, 'f24.png'))
	NN = torch.zeros(24, 1, 28, 28)
	for i in range(0,24):
		NN[i] = nearest_gt(fake_images[i])
		print(i)

	save_image(NN.data, os.path.join(sample_dir, 'NN24.png'))
