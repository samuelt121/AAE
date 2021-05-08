import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.datasets import MNIST
from model_classify import Encoder_FC, Discriminator_FC, Decoder_FC, initialize_weights, Encoder_CNN, Discriminator_CNN, Decoder_CNN

import numpy as np
import config
from utils import load_checkpoint, save_checkpoint, save_latent_distribution, save_reconstructed_images, get_digit_distribution, plot_batch_latent_distribution, check_accuracy
import random

# Load data
train_Dataset = MNIST(root="dataset/", train=True, transform=config.Mytransforms, download=True)
train_Loader = DataLoader(train_Dataset, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=True)
test_Dataset = MNIST(root="dataset/", train=False, transform=config.Mytransforms, download=True)
test_Loader = DataLoader(test_Dataset, shuffle=True, batch_size=100, pin_memory=True)


# Mode - Semi-Supervised
latent_dim = config.latent_space_dim + config.NUM_CLASSES + 1

# Model
if True:  # use FC architectures
    gen = Encoder_FC().to(config.device)
    disc = Discriminator_FC(latent_dim).to(config.device)
    decoder = Decoder_FC().to(config.device)
    dir_string = "FC_SSAAE"
    saved_model_name = 'SSAEE_FC_checkpoint.pth.tar'
    writer_parameters = SummaryWriter(f"logs/SSAAE/FC/parameters")
else:  # use CNN architecture
    gen = Encoder_CNN().to(config.device)
    disc = Discriminator_CNN(latent_dim).to(config.device)
    decoder = Decoder_CNN().to(config.device)
    dir_string = "CNN_SSAAE"
    saved_model_name = 'SSAEE_CNN_checkpoint.pth.tar'
    writer_parameters = SummaryWriter(f"logs/SSAAE/CNN/parameters")

initialize_weights(gen)
initialize_weights(disc)
initialize_weights(decoder)

# Optimizer
opt_gen = optim.Adam(params=gen.parameters(), lr=0.0002, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
opt_disc = optim.Adam(params=disc.parameters(), lr=0.0002, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
opt_decoder = optim.Adam(params=decoder.parameters(), lr=0.0001,
                         betas=(config.ADAM_BETA1, config.ADAM_BETA2))

scheduler_gen = optim.lr_scheduler.MultiStepLR(opt_gen, gamma=0.316, milestones=[200, 1000])  # 0.316 = sqrt(0.1)
scheduler_disc = optim.lr_scheduler.MultiStepLR(opt_disc, gamma=0.316, milestones=[200, 1000])
scheduler_decoder = optim.lr_scheduler.MultiStepLR(opt_decoder, gamma=0.316, milestones=[200, 1000])

# Loss
loss_BCE = nn.BCELoss()
loss_MSE = nn.MSELoss()
loss_classify = nn.CrossEntropyLoss()

# Tensor-board parameters
my_epoch = 0
TB_step = torch.tensor(0)
isSaved = False  # save once in every chosen num of epochs
unlabeled_ratio = 0.8
# Load pre-trained model
load_model = True
try:
    if load_model:
        my_epoch, TB_step = load_checkpoint(torch.load(saved_model_name), gen, disc, decoder, opt_gen, opt_disc, opt_decoder,
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

    for batch_idx, (image, label) in enumerate(train_Loader):
        image = image.to(config.device)
        label = label.to(config.device)

        this_batch_size = len(image)
        hot_vector = np.zeros((this_batch_size, config.NUM_CLASSES + 1), dtype=np.float32)
        idx_unlabeled = random.sample(range(0, this_batch_size), int(this_batch_size * unlabeled_ratio))
        idx_labeled = list(set(range(0, this_batch_size)) - set(idx_unlabeled))
        hot_vector[idx_unlabeled, -1] = 1
        np_label = label.cpu().numpy()
        hot_vector[idx_labeled, np_label[idx_labeled]] = 1

        hot_vector = torch.tensor(hot_vector).unsqueeze(2).unsqueeze(3).to(config.device)
        label[idx_unlabeled] = torch.randint(0, 10, (int(this_batch_size * unlabeled_ratio),)).to(config.device)

        # Reconstruction training + Classifier training
        latentCode = gen(image)  # same as fake_latentCode, with changed
        dec_output, labels_softVector = decoder(latentCode)
        loss_AE = loss_MSE(dec_output, image)
        loss_classifier = loss_classify(labels_softVector, label)  # this loss includes softmax.

        decoder.zero_grad()
        gen.zero_grad()

        loss_AE.backward(retain_graph=True)
        loss_classifier.backward()
        opt_gen.step()
        opt_decoder.step()


        # # ----------------------------------
        # # Reconstruction training
        # latentCode = gen(image)  # same as fake_latentCode, with changed
        # dec_output, _ = decoder(latentCode)
        # loss_AE = loss_MSE(dec_output, image)
        #
        # decoder.zero_grad()
        # gen.zero_grad()
        #
        # loss_AE.backward(retain_graph=True)
        # opt_gen.step()
        # opt_decoder.step()
        #
        # # ----------------------------------
        # # Classifier training:
        # for params in decoder.net.parameters():
        #     params.requires_grad = False
        #
        # for params in gen.parameters():
        #     params.requires_grad = False
        #
        # latentCode = gen(image)  # same as fake_latentCode, with changed
        # _, labels_softVector = decoder(latentCode)
        # loss_classifier = loss_classify(labels_softVector, label)  # this loss includes softmax.
        #
        # decoder.zero_grad()
        # loss_classifier.backward()
        # opt_decoder.step()
        #
        # for params in decoder.net.parameters():
        #     params.requires_grad = True
        #
        # for params in gen.parameters():
        #     params.requires_grad = True

        # ----------------------------------
        # Adversarial training:

        # train Discriminator:
        fake_latentCode = gen(image)
        samples_new = get_digit_distribution(label, num_elements=this_batch_size)
        # plot_batch_latent_distribution(samples_new, label)  # verify p(x) distribution.
        real_latentCode = torch.tensor(samples_new, dtype=torch.float32).unsqueeze(2).unsqueeze(3).to(config.device)

        disc_real = disc(torch.cat((real_latentCode, hot_vector), 1)).view(-1)
        loss_disc_real = loss_BCE(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(torch.cat((fake_latentCode, hot_vector), 1)).view(-1)
        loss_disc_fake = loss_BCE(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # ----------------------------------
        # Train Generator - Adversarial
        fake_latentCode = gen(image)
        disc_fake = disc(torch.cat((fake_latentCode, hot_vector), 1)).view(-1)
        loss_gen = loss_BCE(disc_fake, torch.ones_like(disc_fake))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 225 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] Batch {batch_idx}/{len(train_Loader)} \
                  Loss Disc: {loss_disc:.4f}, loss G: {loss_gen:.4f}, loss AE: {loss_AE:.4f},"
                f" Loss Classifier: {loss_classifier:.4f}")

            if epoch % 3 == 0:
                # Plot things to tensor-board
                with torch.no_grad():
                    accuracy = check_accuracy(test_Loader, gen, decoder)
                    writer_parameters.add_scalar('Discriminator loss', loss_disc, global_step=TB_step)
                    writer_parameters.add_scalar('Generator loss', loss_gen, global_step=TB_step)
                    writer_parameters.add_scalar('AutoEncoder loss', loss_AE, global_step=TB_step)
                    writer_parameters.add_scalar('Classifier loss', loss_classifier, global_step=TB_step)
                    writer_parameters.add_scalar('Test-set accuracy', accuracy, global_step=TB_step)

                    TB_step += 1

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

    isSaved = False
    scheduler_gen.step()
    scheduler_disc.step()
    scheduler_decoder.step()
