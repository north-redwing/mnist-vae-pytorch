import argparse
import numpy as np
import torch
import torch.utils as utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_loader(path_to_dataset, batch_size, num_workers):
    """Get a train loader and evaluation loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_datasets = datasets.MNIST(
        root=path_to_dataset,
        train=True,
        download=True,
        transform=transform
    )
    eval_datasets = datasets.MNIST(
        root=path_to_dataset,
        train=False,
        download=True,
        transform=transform
    )
    train_loader = utils.data.DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = utils.data.DataLoader(
        eval_datasets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, eval_loader


def plot_results(model, inputs, reconstructions, latent_variables, height,
                 width):
    model.eval()
    fig = plt.figure(figsize=(20, 6))

    inputs = inputs.view(-1, height, width)
    reconstructions = reconstructions.view(-1, height, width)
    n_col = latent_variables.shape[1]

    mask = np.random.random_integers(low=0, high=inputs.shape[0] - 1, size=n_col)
    latent_variables = latent_variables[mask]
    inputs = inputs[mask]
    reconstructions = reconstructions[mask]

    # plot original images
    for idx, image in enumerate(inputs.detach().numpy()[:n_col]):
        ax = fig.add_subplot(3, n_col, idx + 1, xticks=[], yticks=[])
        ax.imshow(image, 'gray')

    # plot reconstructed images
    for idx, image in enumerate(reconstructions.cpu().detach().numpy()[:n_col]):
        ax = fig.add_subplot(3, n_col, idx + (n_col + 1), xticks=[], yticks=[])
        ax.imshow(image, 'gray')

        # plot trainsition from original images to reconstructed images
    transition = torch.cat([
        latent_variables[1, :] * (i * 0.1)
        + latent_variables[0, :] * ((10 - i) * 0.1)
        for i in range(n_col)
    ])
    with torch.no_grad():
        transition = transition.reshape(n_col, n_col)
        transition = model.decoder(transition).view(-1, height, width)
    for idx, image in enumerate(transition.cpu().detach().numpy()[:n_col]):
        ax = fig.add_subplot(3, n_col, idx + (2 * n_col + 1), xticks=[],
                             yticks=[])
        ax.imshow(image, 'gray')

    return fig


def project_results(latent_variables, inputs, labels, writer, epoch, height, width):
    inputs = inputs.reshape(-1, 1, height, width)
    mask = np.random.random_integers(low=0, high=inputs.shape[0] - 1, size=500)
    latent_variables = latent_variables[mask]
    inputs = inputs[mask]
    labels = labels[mask]
    writer.add_embedding(latent_variables, metadata=labels,
                         label_img=inputs, global_step=epoch)


def get_args():
    parser = argparse.ArgumentParser(description='VAE')

    parser.add_argument('--batch_size', type=int, default=216,
                        help='input batch size (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='number of epochs (default: 20)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed value (default: 0)')
    parser.add_argument('--no_hyperdash', action='store_true', default=False,
                        help='disables Hyperdash logging')
    parser.add_argument('--checkpoint_dir_name', type=str, default=None,
                        help='resume training by using the trained model of checkpoint')

    args = parser.parse_args()

    # CUDA setting
    args.device = torch.device('cpu')
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        if torch.cuda.device_count() > 1:
            gpu_ids = [id for id in range(len(torch.cuda.device_count()))]
            args.device = torch.device(f'cuda:{gpu_ids[0]}')
    print('####################')
    print('device ===> ', args.device)
    print('####################')

    return args
