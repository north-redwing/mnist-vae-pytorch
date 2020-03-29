import os
import datetime
import csv
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pathlib import Path
from time import time
from utils import get_args, get_loader, plot_results, project_results
from net import VAE
import warnings


def train(model, train_loader, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        model(inputs)
        loss = model.loss_function(inputs) / batch_size
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss


def eval(epoch, model, eval_loader, device, writer, height, width, batch_size):
    model.eval()
    running_loss = 0.0
    inputs_epoch, labels_epoch, latent_variables_epoch, reconstructions_epoch = \
        torch.Tensor(), torch.LongTensor(), torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            reconstructions, latent_variables = model(inputs)
            loss = model.loss_function(inputs) / batch_size
            running_loss += loss.item()
            inputs_epoch = torch.cat([
                inputs_epoch,
                inputs.cpu().detach()
            ], dim=0
            )
            labels_epoch = torch.cat([
                labels_epoch,
                labels.cpu().detach()
            ], dim=0
            )
            latent_variables_epoch = torch.cat([
                latent_variables_epoch,
                latent_variables.cpu().detach()
            ], dim=0
            )
            reconstructions_epoch = torch.cat([
                reconstructions_epoch,
                reconstructions.cpu().detach()
            ], dim=0
            )
    fig = plot_results(
        model,
        inputs,
        reconstructions,
        latent_variables,
        height,
        width
    )
    writer.add_figure('results', fig, global_step=epoch)
    project_results(
        latent_variables_epoch,
        inputs_epoch,
        labels_epoch,
        writer,
        epoch,
        height,
        width
    )
    eval_loss = running_loss / len(eval_loader)
    return eval_loss


def main():
    warnings.filterwarnings('ignore')
    start_time = time()
    args = get_args()
    if args.checkpoint_dir_name:
        dir_name = args.checkpoint_dir_name
    else:
        dir_name = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    path_to_dir = Path(__file__).resolve().parents[1]
    path_to_dir = os.path.join(path_to_dir, *['log', dir_name])
    os.makedirs(path_to_dir, exist_ok=True)
    # tensorboard
    path_to_tensorboard = os.path.join(path_to_dir, 'tensorboard')
    os.makedirs(path_to_tensorboard, exist_ok=True)
    writer = SummaryWriter(path_to_tensorboard)
    # model saving
    os.makedirs(os.path.join(path_to_dir, 'model'), exist_ok=True)
    path_to_model = os.path.join(path_to_dir, *['model', 'model.tar'])
    # csv
    os.makedirs(os.path.join(path_to_dir, 'csv'), exist_ok=True)
    path_to_results_csv = os.path.join(path_to_dir, *['csv', 'results.csv'])
    path_to_args_csv = os.path.join(path_to_dir, *['csv', 'args.csv'])
    if not args.checkpoint_dir_name:
        with open(path_to_args_csv, 'a') as f:
            args_dict = vars(args)
            param_writer = csv.DictWriter(f, list(args_dict.keys()))
            param_writer.writeheader()
            param_writer.writerow(args_dict)

    # logging using hyperdash
    if not args.no_hyperdash:
        from hyperdash import Experiment
        exp = Experiment('Generalization task on CIFAR10 dataset with VAE')
        for key in vars(args).keys():
            exec("args.%s = exp.param('%s', args.%s)" % (key, key, key))
    else:
        exp = None

    path_to_dataset = os.path.join(
        Path(__file__).resolve().parents[2],
        'datasets'
    )
    os.makedirs(path_to_dataset, exist_ok=True)
    train_loader, eval_loader = get_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        path_to_dataset=path_to_dataset
    )

    # define a network, loss function and optimizer
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    model = VAE(x_dim=28 * 28, z_dim=10, device=args.device)
    torch.jit.trace(model, images, check_trace=False)
    writer.add_graph(model, images)
    # model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    # resume training
    if args.checkpoint_dir_name:
        print('\nLoading the model...')
        checkpoint = torch.load(path_to_model)
        model.state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    summary(model, input_size=(1, 28, 28))
    model.to(args.device)

    # train the network
    print('\n--------------------')
    print('Start training and evaluating the VAE')
    for epoch in range(start_epoch, args.n_epoch):
        start_time_per_epoch = time()
        train_loss = train(
            model,
            train_loader,
            optimizer,
            args.device,
            args.batch_size
        )
        eval_loss = eval(
            epoch,
            model,
            eval_loader,
            args.device,
            writer,
            28,
            28,
            args.batch_size
        )
        elapsed_time_per_epoch = time() - start_time_per_epoch
        result_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'elapsed time': elapsed_time_per_epoch
        }
        with open(path_to_results_csv, 'a') as f:
            result_writer = csv.DictWriter(f, list(result_dict.keys()))
            if epoch == 0: result_writer.writeheader()
            result_writer.writerow(result_dict)
        # checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            path_to_model
        )
        if exp:
            exp.metric('train loss', train_loss)
            exp.metric('eval loss', eval_loss)
        else:
            print(result_dict)

        writer.add_scalar(
            'loss/train_loss', train_loss,
            epoch * len(train_loader)
        )
        writer.add_scalar(
            'loss/eval_loss',
            eval_loss,
            epoch * len(eval_loader)
        )

    elapsed_time = time() - start_time
    print('\nFinished Training, elapsed time ===> %f' % elapsed_time)
    if exp:
        exp.end()
    writer.close()


if __name__ == '__main__':
    main()
