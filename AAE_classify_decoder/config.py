import torchvision.transforms as transforms
import torch

# General parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# parameters for training MNIST dataset
N_channels = 1
IMAGE_SIZE = 28
BATCH_SIZE = 128
NUM_EPOCHS = 5000
NUM_CLASSES = 10
DISCRIMINATOR_ITERATIONS = 1  # for every Generator training iterations.

TEST_BATCH_SIZE = 100

# Latent space
latent_space_dim = 2


# Optimizer parameters
LEARNING_RATE = 3e-4
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.99


# Data Loading parameters
Mytransforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float32),
    # transforms.Normalize([0.5 for _ in range(N_channels)], [0.5 for _ in range(N_channels)]),  # values of MNIST dataset: (0.1307, 0.3081)
])


#
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'xkcd:sky blue', 'tab:pink', 'tab:orange']
# https://matplotlib.org/stable/api/colors_api.html?highlight=colors#module-matplotlib.colors

