import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import torchvision
from torchvision.datasets import MNIST
from model import Encoder_FC, Discriminator_FC, Decoder_FC, initialize_weights, Encoder, Discriminator, Decoder

import os
import config
from utils import load_checkpoint, save_checkpoint, save_latent_distribution, save_reconstructed_images, plot_batch_latent_distribution
import random

# Load data
train_Dataset = MNIST(root="dataset/", train=True, transform=config.Mytransforms, download=True)
train_Loader = DataLoader(train_Dataset, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=True)
test_Dataset = MNIST(root="dataset/", train=False, transform=config.Mytransforms, download=True)
test_Loader = DataLoader(test_Dataset, shuffle=True, batch_size=100, pin_memory=True)


# Mode - Unsupervised
latent_dim = config.latent_space_dim

# Model
if True:  # use FC architectures
    gen = Encoder_FC().to(config.device)
    disc = Discriminator_FC(latent_dim).to(config.device)
    decoder = Decoder_FC().to(config.device)
    dir_string = "FC_AAE"
    saved_model_name = "AEE_FC_checkpoint.pth.tar"
else:  # use CNN architecture
    gen = Encoder().to(config.device)
    disc = Discriminator(latent_dim).to(config.device)
    decoder = Decoder().to(config.device)
    dir_string = "CNN_AAE"
    saved_model_name = 'AEE_CNN_checkpoint.pth.tar'

initialize_weights(gen)
initialize_weights(disc)
initialize_weights(decoder)

# Optimizer
opt_gen = optim.Adam(params=gen.parameters(), lr=0.0002, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
opt_disc = optim.Adam(params=disc.parameters(), lr=0.0002, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
opt_decoder = optim.Adam(params=decoder.parameters(), lr=3e-4,
                         betas=(config.ADAM_BETA1, config.ADAM_BETA2))

scheduler_gen = optim.lr_scheduler.MultiStepLR(opt_gen, gamma=0.316, milestones=[100, 1000])  # 0.316 = sqrt(0.1)
scheduler_disc = optim.lr_scheduler.MultiStepLR(opt_disc, gamma=0.316, milestones=[100, 1000])
scheduler_decoder = optim.lr_scheduler.MultiStepLR(opt_decoder, gamma=0.316, milestones=[100, 1000])

# Loss
loss_BCE = nn.BCELoss()
loss_MSE = nn.MSELoss()

# General parameters
my_epoch = 0
TB_step = torch.tensor(0)
isSaved = False  # save once in every chosen num of epochs

# latent distribution
mu = torch.zeros(config.latent_space_dim)
cov = 25*torch.eye(config.latent_space_dim)  # sigma = sqrt(5)
prior = torch.distributions.MultivariateNormal(mu, cov)

# Load pre-trained model
load_model = True
try:
    if load_model:
        my_epoch = load_checkpoint(torch.load(saved_model_name), gen, disc, decoder, opt_gen, opt_disc, opt_decoder,
                                      scheduler_gen, scheduler_disc, scheduler_decoder)
except FileNotFoundError:
    print("Model file doesn't exist")

# --------------------------------
# --------Training ---------------

gen.train()
disc.train()
decoder.train()

# start Train loop
for epoch in range(my_epoch, config.NUM_EPOCHS):
    # save checkpoint
    if epoch % 10 == 0 or epoch == config.NUM_EPOCHS - 1:
        checkpoint = {'gen_state_dict': gen.state_dict(), 'gen_optimizer': opt_gen.state_dict(),
                      'disc_state_dict': disc.state_dict(), 'disc_optimizer': opt_disc.state_dict(),
                      'decoder_state_dict': decoder.state_dict(), 'decoder_optimizer': opt_decoder.state_dict(),
                      'epoch_num': epoch, 'TB_step': TB_step, 'decoder_scheduler': scheduler_decoder.state_dict(),
                      'gen_scheduler': scheduler_gen.state_dict(), 'disc_scheduler': scheduler_disc.state_dict()}

        save_checkpoint(checkpoint, filename=saved_model_name)

    for batch_idx, (image, _) in enumerate(train_Loader):
        image = image.to(config.device)

        # ----------------------------------
        # Reconstruction training:
        latentCode = gen(image)  # same as fake_latentCode, with changed
        dec_output = decoder(latentCode)
        loss_AE = loss_MSE(dec_output, image)

        decoder.zero_grad()
        gen.zero_grad()

        loss_AE.backward()
        opt_gen.step()
        opt_decoder.step()

        # ----------------------------------
        # Adversarial training:

        # train Discriminator:
        fake_latentCode = gen(image)
        real_latentCode = prior.sample((config.BATCH_SIZE,)).unsqueeze(2).unsqueeze(3).to(config.device)
        # real_latentCode = torch.randn((config.BATCH_SIZE, config.latent_space_dim)).unsqueeze(2).unsqueeze(3).to(
        #     config.device)
        disc_real = disc(real_latentCode).view(-1)
        loss_disc_real = loss_BCE(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_latentCode).view(-1)
        loss_disc_fake = loss_BCE(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        fake_latentCode = gen(image)
        disc_fake = disc(fake_latentCode).view(-1)
        loss_gen = loss_BCE(disc_fake, torch.ones_like(disc_fake))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 225 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] Batch {batch_idx}/{len(train_Loader)} \
                  Loss Disc: {loss_disc:.4f}, loss G: {loss_gen:.4f}, loss AE: {loss_AE:.4f}"
            )

            if epoch % 10 == 0 and not isSaved:
                isSaved = True
                with torch.no_grad():
                    img_orig_image = torchvision.utils.make_grid(image, normalize=True)
                    img_orig_image = img_orig_image.permute(1, 2, 0).detach().cpu().numpy()
                    img_reconstructed_image = torchvision.utils.make_grid(dec_output, normalize=True)
                    img_reconstructed_image = img_reconstructed_image.permute(1, 2, 0).detach().cpu().numpy()

                    # print('save figures')
                    save_reconstructed_images(img_orig_image, img_reconstructed_image, epoch, dir_string)
                    save_latent_distribution(test_Loader, gen, epoch, dir_string)

                    # Plot things to tensor-board
                    config.writer_parameters.add_scalar('Discriminator loss', loss_disc, global_step=TB_step)
                    config.writer_parameters.add_scalar('Generator loss', loss_gen, global_step=TB_step)
                    config.writer_parameters.add_scalar('AutoEncoder loss', loss_AE, global_step=TB_step)

                    TB_step += 1

    isSaved = False
    scheduler_gen.step()
    scheduler_disc.step()
    scheduler_decoder.step()

def check_accuracy(loader, gen, decoder):
    pass