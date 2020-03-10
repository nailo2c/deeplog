import argparse
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train(args):
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        logger.info('Use CUDA')
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader('train', args.batch_size, args.window_size, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    model = Model(args.input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for seq, label in train_loader:
            seq = seq.clone().detach().view(-1, args.window_size, args.input_size).to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logger.debug('Epoch [{}/{}], Train_loss: {}'.format(
            epoch, args.epochs, round(train_loss/len(train_loader.dataset), 4)
        ))
    logger.debug('Finished Training')
    save_model(model, args.model_dir, args)


def _get_train_data_loader(name, batch_size, window_size, **kwargs):
    logger.info("Get train data loader")
    seq_dataset = _generate(name, window_size)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True,
                            sampler=None, **kwargs)
    return dataloader


def _generate(name, window_size=10):
    num_sessions = 0
    inputs = []
    outputs = []

    with open(name, 'r') as f:
        for line in f.readlines():
            # line = line.decode().rstrip()  # decode from byte to string & right strip \n
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i+window_size])
                outputs.append(line[i+window_size])
            num_sessions += 1
    logger.info('Number of session({}): {}'.format(name, len(inputs)))
    logger.info('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def save_model(model, model_dir, args):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    # Save arguments used to create model for restoring the model later
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_classes': args.num_classes,
            'num_candidates': args.num_candidates,
            'window_size': args.window_size,
        }
        torch.save(model_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--window-size', type=int, default=10, metavar='N',
                        help='length of training window (default: 10)')
    parser.add_argument('--input-size', type=int, default=1, metavar='N',
                        help='model input size (default: 1)')
    parser.add_argument('--hidden-size', type=int, default=64, metavar='N',
                        help='hidden layer size (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='number of model\'s layer (default: 2)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-gpus', type=int, default=0,
                        helper='number of gpu to train')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        helper='the place where to store the training data.')
    parser.add_argument('--model-dir', type=str, default="./model/",
                        help='the place where to store the model parameter.')

    parser.add_argument('--num-classes', type=int, metavar='N',
                        help='the number of model\'s output, must same as pattern size!')
    parser.add_argument('--num-candidates', type=int, metavar='N',
                        help='the number of predictors sequences as correct predict.')

    train(parser.parse_args())
