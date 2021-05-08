import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
import config


def save_checkpoint(state, filename):
    print('----> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model_gen, model_disc, model_decoder, optimizer_gen, optimizer_disc, optimizer_decoder,
                    gen_scheduler, disc_scheduler, decoder_scheduler):
    print("----> Loading checkpoint")
    model_gen.load_state_dict(checkpoint['gen_state_dict'])
    optimizer_gen.load_state_dict(checkpoint['gen_optimizer'])
    model_disc.load_state_dict(checkpoint['disc_state_dict'])
    optimizer_disc.load_state_dict(checkpoint['disc_optimizer'])
    model_decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer_decoder.load_state_dict(checkpoint['decoder_optimizer'])
    gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
    disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
    decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler'])

    return checkpoint['epoch_num']  #, checkpoint['TB_step']


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = (channels_sum / num_batches)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) ** 0.5 * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
z = gaus2d(x, y, mx=1, my=0, sx=1, sy=0.2)


def get_digit_distribution(digits, num_elements):
    rad_angles = [2 * np.pi * float(digit / config.NUM_CLASSES) for digit in digits]
    Rotation_matrices = [[[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in
                         rad_angles]
    shifts = [[np.cos(angle), - np.sin(angle)] for angle in rad_angles]

    # create simple non-symmetric gaussian
    mx, my = 0, 0
    sx, sy = 1.5, 0.2
    mean = [mx, my]
    cov = [[sx ** 2, 0], [0, sy ** 2]]

    samples = np.random.multivariate_normal(mean, cov, (num_elements,))
    shift = np.multiply(shifts, [3 * max(sx, sy), 3 * max(sx, sy)])

    samples_new = [shift[idx] + np.matmul(samples[idx], Rotation_matrices[idx]) for idx in range(num_elements)]

    return samples_new


def save_reconstructed_images(orig_img, reconstructed_img, epoch_num, mode):
    plt.ioff()

    fig, ax = plt.subplots()
    fig.suptitle(f"Epoch {epoch_num}")
    ax1 = fig.add_subplot(121)
    ax1.imshow(orig_img)
    ax1.title.set_text('Original')
    ax2 = fig.add_subplot(122)
    ax2.imshow(reconstructed_img)
    ax2.title.set_text('Reconstructed')
    fig.savefig("results/" + mode + "/reconstructed_image/epoch_" + str(epoch_num) + ".jpg")
    plt.close(fig)


def save_latent_distribution(loader, encoder, epoch_num, mode):
    samples_distribution = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for image, label in loader:
            image = image.to(config.device)
            label = label.to(config.device)

            latent_vector = encoder(image).squeeze(3).squeeze(2).cpu().numpy()
            samples_distribution = np.vstack([samples_distribution, latent_vector]) if samples_distribution.size else latent_vector
            labels = np.hstack([labels, label.cpu().numpy()]) if labels.size else label.cpu().numpy()
        plt.ioff()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(f"Epoch {epoch_num}")
        data_colors = [config.colors[digit] for digit in labels]
        x = samples_distribution[:, 0]
        y = samples_distribution[:, 1]

        ax.scatter(x, y, c=data_colors)

        # create legend
        legend_dict = {str(i): config.colors[i] for i in range(config.NUM_CLASSES)}
        patchList = [mpatches.Patch(color=legend_dict[key], label=key) for key in legend_dict]
        plt.legend(handles=patchList)
        plt.title('Latent Space Distribution')
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-4, 4)
        fig.savefig("results/" + mode + "/latent_distribution/epoch_" + str(epoch_num) + ".jpg")
        plt.close(fig)


def plot_latent_distribution(loader, encoder):
    samples_distribution = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for image, label in loader:
            image = image.to(config.device)
            label = label.to(config.device)

            latent_vector = encoder(image).squeeze(3).squeeze(2).cpu().numpy()
            samples_distribution = np.vstack([samples_distribution, latent_vector]) if samples_distribution.size else latent_vector
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
        plt.show()


def plot_gmm(num_classes=config.NUM_CLASSES):
    import matplotlib.pyplot as plt
    num_classes = 10
    rad_angles = [2 * np.pi * float(digit / num_classes) for digit in range(num_classes)]
    Rotation_matrices = [[[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in rad_angles]
    shifts = [[np.cos(angle), - np.sin(angle)] for angle in rad_angles]

    mx, my = 0, 0
    sx, sy = 1.5,  0.25

    mean = [mx, my]
    cov = [[sx ** 2, 0], [0, sy ** 2]]
    x = np.random.multivariate_normal(mean, cov, (1000,))

    x_total = np.array(())

    for digit in range(num_classes):
        Rotation_matrix = Rotation_matrices[digit]
        shift = np.multiply(shifts[digit], [5*max(sx, sy), 5*max(sx, sy)])

        x_new = shift + np.matmul(x, Rotation_matrix)
        x_total = np.vstack([x_total, x_new]) if x_total.size else x_new
    # plot gmm
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_total[:, 0], x_total[:, 1], color='b', marker='o')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    # plt.title(f"correlation equals {digit}")
    plt.show()


def plot_batch_latent_distribution(samples, labels):
    txt = labels.cpu().numpy()
    samples_new = np.array(samples)
    fig, ax = plt.subplots()
    x = samples_new[:, 0]
    y = samples_new[:, 1]
    ax.scatter(x, y)

    for i, lab in enumerate(txt):
        ax.annotate(lab, (x[i], y[i]))