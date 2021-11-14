import random
random.seed(101)
import itertools
import torch
import time
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data import Data
from parse import parse_args
from model import BPRMF, BCEMF


def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0,

    # Find latest epoch
    epoch = 0
    for i in itertools.count(1):
        if 'epoch={}.checkpoint.pth.tar'.format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print("Which epoch to load from? Choose in range [0, {}]."
              .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename)

    try:
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
              .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch


def restore_best_checkpoint(epoch, model, checkpoint_dir):
    """
    Restore the best performance checkpoint
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    print("=> Successfully restored checkpoint (trained for {} epochs)"
          .format(checkpoint['epoch']))

    return model


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch):
    # Evaluate validation dataset

    eval_valid = ProxyEvaluator(data, data.train_user_list, data.test_user_list, top_k=[20])

    ret, _ = eval_valid.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[4]}

    perf_str = 'Validation: recall={}, ' \
               'precision={}, hit={}, ndcg={}'.format(str(n_ret["recall"]),
                                                      str(n_ret['precision']), str(n_ret['hit_ratio']),
                                                      str(n_ret['ndcg']))
    print(perf_str)
    with open('stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(perf_str + "\n")

    # Check if need to early stop
    if ret[1] > data.best_valid_recall:
        data.best_valid_epoch = epoch
        data.best_valid_recall = ret[1]
        data.patience = 0
    else:
        data.patience += 1
        if data.patience >= args.patience:
            print_str = "The best performance epoch is % d " % data.best_valid_epoch
            print(print_str)
            return True

    return False


def result(args, data, model):
    eval_test= ProxyEvaluator(data, data.train_user_list, data.test_user_list)

    ret, _ = eval_test.evaluate(model)

    n_ret = {"precision":ret[0],"recall": ret[1],"MAP":ret[2],"NDCG":ret[3],"MRR":ret[4],"hit_ratio": ret[5]}

    perf_str = 'tests: {}'.format(n_ret)

    with open('stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(perf_str + "\n")


if __name__ == '__main__':

    # Start training
    start = time.time()

    args = parse_args()
    data = Data(args)
    data.load_data()

    if args.modeltype == 'BPFMF':
        model = BPRMF(args, data)
    if args.modeltype == 'BCEMF':
        model = BCEMF(args, data)

    model.cuda(0)

    model, start_epoch = restore_checkpoint(model, args.checkpoint)

    model.train()

    n_batch = data.n_observations // args.batch_size + 1

    flag = False

    # Training
    for epoch in range(start_epoch, args.epoch):

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            model = restore_best_checkpoint(data.best_valid_epoch, model, args.checkpoint)
            break

        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        # Running through several batches of data
        for idx in tqdm(range(n_batch)):
            # Sample batch-sized data from training dataset
            users, pos_items, neg_items = data.sample()

            # Get the slice of embedded data and convert to GPU
            users = model.embed_user(torch.tensor(users).cuda(0))
            pos_items = model.embed_item(torch.tensor(pos_items).cuda(0))
            neg_items = model.embed_item(torch.tensor(neg_items).cuda(0))

            optimizer.zero_grad()

            mf_loss, reg_loss = model(users, pos_items, neg_items)

            loss = mf_loss + reg_loss

            loss.backward()

            optimizer.step()

            running_loss += loss.detach().item()

            running_mf_loss += mf_loss.detach().item()

            running_reg_loss += reg_loss.detach().item()

            num_batches += 1

        # Training data for one epoch
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            epoch, time.time() - start, running_loss / num_batches,
            running_mf_loss / num_batches, running_reg_loss / num_batches)

        print(perf_str)

        with open('stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(perf_str + "\n")

        # Save checkpoints
        save_checkpoint(model, epoch + 1, args.checkpoint)

        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0:
            flag = evaluation(args, data, model, epoch)

    # Get result
    result(args, data, model)