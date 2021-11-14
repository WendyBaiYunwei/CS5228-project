import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yelp2018.new',
                        help='Choose a dataset')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--saveID', type=int, default=1,
                        help='Specify model save path.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')
    parser.add_argument('--checkpoint', type=str, default='./Yelp2018',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'BPFMF',
                        help='Specify model save path.')
    return parser.parse_args()