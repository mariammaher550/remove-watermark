import dataset
import discriminator_model 
import generator_model
import run_on_patches
import vgg_loss
import numpy as np

# Set random seed
np.random.seed(42)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# The GAN
D = discriminator_model.Discriminator()
G = generator_model.Generator()
G.