import os
import sys
import logging
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deeplog.deeplog import train

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

    parser.add_argument('--num-classes', type=int, metavar='N',
                        help='the number of model\'s output, must same as pattern size!')
    parser.add_argument('--num-candidates', type=int, metavar='N',
                        help='the number of predictors sequences as correct predict.')

    # Pass container environment
    parser.add_argument('--hosts', type=list, default=['127.0.0.1'],
                        help='args for SageMaker distributed training.')
    parser.add_argument('--current-host', type=str, default='127.0.0.1',
                        help='args for SageMaker distributed training.')
    parser.add_argument('--model-dir', type=str, default='./model/',
                        help='the place where to store the model parameter.')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='the place where to store the training data.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpu to train')

    # Local mode
    parser.add_argument('--local', action='store_true',
                        help='use local training files instead of S3.')

    if not os.path.isdir('./model/'):
        os.mkdir('./model/')
    train(parser.parse_args())
