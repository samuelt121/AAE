import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter

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


# Tensor-board plots
writer_parameters = SummaryWriter(f"logs/parameters")

#
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'xkcd:sky blue', 'tab:pink', 'tab:orange']
# https://matplotlib.org/stable/api/colors_api.html?highlight=colors#module-matplotlib.colors



# other chedulers.
# scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, factor=0.9, patience=4, verbose=True)
# scheduler_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: 0.65 ** epoch)
# scheduler_gen = torch.optim.lr_scheduler.CyclicLR(opt_gen, base_lr=config.LEARNING_RATE, max_lr=1.5*config.LEARNING_RATE,
#                                                  step_size_up=15, mode="triangular2", cycle_momentum=False)
