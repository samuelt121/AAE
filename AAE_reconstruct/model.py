"""
Script containing the implementation of the U-NET
"""
import torch
import torch.nn as nn
from torchsummary import summary
import config

# ---------------------------------------------------
# Fully Connected networks, as suggested by article.
# ---------------------------------------------------


class Encoder_FC(nn.Module):
    def __init__(self, layerDepth = 1000):
        super(Encoder_FC, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            self._Block(config.IMAGE_SIZE * config.IMAGE_SIZE * config.N_channels, layerDepth),
            self._Block(layerDepth, layerDepth),
            nn.Linear(layerDepth, config.latent_space_dim),
            nn.Unflatten(1, (config.latent_space_dim, 1, 1))
        )

    def _Block(self, input_dim, output_dim):
        return nn.Sequential(
                   nn.Linear(input_dim, output_dim),
                   nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Decoder_FC(nn.Module):
    def __init__(self, layerDepth = 1000):
        super(Decoder_FC, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            self._Block(config.latent_space_dim, layerDepth),
            self._Block(layerDepth, layerDepth),
            nn.Linear(layerDepth, config.IMAGE_SIZE * config.IMAGE_SIZE * config.N_channels),
            nn.Sigmoid(),
            nn.Unflatten(1, (config.N_channels, config.IMAGE_SIZE, config.IMAGE_SIZE))
        )

    def _Block(self, input_dim, output_dim):
        return nn.Sequential(
                   nn.Linear(input_dim, output_dim),
                   nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)
        # decoded_image = self.net(x)
        # decoded_image = nn.Unflatten(1, (config.N_channels, config.IMAGE_SIZE, config.IMAGE_SIZE))(decoded_image)
        #
        # return decoded_image


class Discriminator_FC(nn.Module):
    def __init__(self, input_dim, layerDepth = 1000):
        super(Discriminator_FC, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            self._Block(input_dim, layerDepth),
            self._Block(layerDepth, layerDepth),
            nn.Linear(layerDepth, 1),
            nn.Sigmoid()
        )

    def _Block(self, input_dim, output_dim):
        return nn.Sequential(
                   nn.Linear(input_dim, output_dim),
                   nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------
# CNN implementation, for better classification.
# ---------------------------------------------------


# Generator is D-CNN network.
# Input: Batch of Images. size:[N x channels_img x Image_size, Image_size] -> Im_size = 28
# Output: latent-space vector [latent_dim x 1]
class Encoder(nn.Module):
    def __init__(self, gen_features=4):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            self._Block(config.N_channels, gen_features, 4, 2, 1),
            self._Block(gen_features, 2 * gen_features, 4, 2, 1),
            self._Block(2 * gen_features, 4 * gen_features, 5, 2, 1), #Im_size: [3x3]
            nn.Flatten(),
            nn.Linear(3*3*4*gen_features, config.latent_space_dim, bias=True),
            nn.Unflatten(1, (config.latent_space_dim, 1, 1))
        )

    def _Block(self, input_dim, output_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_dim),  # Im_size : [7x7]
        )

    def forward(self, x):
        return self.net(x)


# Input: latent-space vector
# Output: Image. size: [28x28]
class Decoder(nn.Module):
    def __init__(self, dec_features=4):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            self._Block(config.latent_space_dim, 4 * dec_features, 3, 1, 0),  # [3x3]
            self._Block(4 * dec_features, 2 * dec_features, 5, 2, 1),  # Im_size : [7x7]
            self._Block(dec_features * 2, dec_features, 4, 2, 1),
            self._Block(dec_features, config.N_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def _Block(self, input_dim, output_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        return self.net(x)


# Input - latent vector
# Output - value (real or fake)
class Discriminator(nn.Module):
    def __init__(self, input_dim, features_disc=1000):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            self._Block(input_dim, features_disc),
            self._Block(features_disc, int(features_disc / 2)),
            self._Block(int(features_disc / 2), int(features_disc / 4)),
            self._Block(int(features_disc / 4), int(features_disc / 8)),
            self._Block(int(features_disc / 8), 1),
            nn.Sigmoid()
        )

    def _Block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    # verify networks' data transition.

    enc = Encoder()
    dec = Decoder()
    disc = Discriminator(2)

    initialize_weights(enc)

    input = torch.randn((100, 1, 28, 28))
    gen_output = enc(input)
    print(gen_output.shape)
    dec_output = dec(gen_output)
    print(dec_output.shape)
    disc_output = disc(gen_output)
    print(disc_output.shape)
