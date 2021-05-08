import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from model_classify import Encoder_FC, Discriminator_FC, Decoder_FC

import config
from utils import load_checkpoint_main, plot_latent_distribution
import cv2
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import time


# ----- Helpful function -----
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    global isClick
    isClick = True

    global coords
    coords.append((ix, iy))

    return coords
# -------------------

# Load Data
test_Dataset = MNIST(root="dataset/", train=False, transform=config.Mytransforms, download=True)
test_Loader = DataLoader(test_Dataset, shuffle=True, batch_size=100, pin_memory=True)

# Load Model
latent_dim = config.latent_space_dim + config.NUM_CLASSES + 1
saved_model_name = 'SSAEE_FC_checkpoint.pth.tar'
encoder = Encoder_FC().to(config.device)
disc = Discriminator_FC(latent_dim).to(config.device)
decoder = Decoder_FC().to(config.device)

try:
    if True:
        load_checkpoint_main(torch.load(saved_model_name), encoder, disc, decoder)
except FileNotFoundError:
    print("Model file doesn't exist")


# Set Windows
# cv2.namedWindow('Latent Space', flags=cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Generated Digit', flags=cv2.WINDOW_KEEPRATIO)

# Plot latent space
samples_distribution = np.array([])
labels = np.array([])

with torch.no_grad():
    for image, label in test_Loader:
        image = image.to(config.device)
        label = label.to(config.device)

        latent_vector = encoder(image).squeeze(3).squeeze(2).cpu().numpy()
        samples_distribution = np.vstack(
            [samples_distribution, latent_vector]) if samples_distribution.size else latent_vector
        labels = np.hstack([labels, label.cpu().numpy()]) if labels.size else label.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    data_colors = [config.colors[digit] for digit in labels]
    x = samples_distribution[:, 0]
    y = samples_distribution[:, 1]

    ax.scatter(x, y, c=data_colors)

    # create legend
    legend_dict = {str(i): config.colors[i] for i in range(config.NUM_CLASSES)}
    patchList = [mpatches.Patch(color=legend_dict[key], label=key) for key in legend_dict]
    plt.legend(handles=patchList)
    plt.title('Latent Space Distribution')
    plt.show(block=False)

isClick = False
coords = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cnt = 0  # helpful for identifying whether user clicked canvas


while True:

    key = cv2.waitKey(25)
    if key == ord('q'):
        fig.canvas.mpl_disconnect(cid)
        break

    plt.ginput(1, show_clicks=True, timeout=0)
    if isClick:
        isClick = False
        with torch.no_grad():
            fake_latent = torch.tensor(coords[-1], dtype=torch.float32).reshape(1, 2, 1, 1).to(config.device)
            restored_digit, _ = decoder(fake_latent)
            restored_digit = restored_digit.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            cv2.imshow('Generated Digit', restored_digit)
