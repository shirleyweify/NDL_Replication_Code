from __future__ import print_function
from preprocess import *
from ndlnet import *
import argparse
import numpy as np
import os
import pandas as pd
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
from scipy import signal
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score

torch.cuda.empty_cache()


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    # input a list of filenames (subjects)

    def __init__(self, list_IDs, filepath, para):
        'Initialization'
        self.list_IDs = list_IDs
        self.filepath = filepath
        self.para = para

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        X = np.load(self.filepath + ID)  # segments
        tot_length = X['eeg'].shape[1]
        sel_length = self.para['T'] + self.para['p']
        data = X['eeg'][:, int(tot_length / 2 - sel_length / 2): int(tot_length / 2 + sel_length / 2)]
        # res = X['eeg'][:, :] - X['eeg'][:, :].mean(0)[np.newaxis, :] # column means
        res = data - data.mean(1)[:, np.newaxis]  # row means
        res = torch.from_numpy(res).unsqueeze(0).float()
        res = res / res.std()
        # target = torch.from_numpy(X['target'][:2]).float()
        # target[1] = target[1] / 0.25
        target = torch.tensor([X['target'][0]])
        return res, target


def train(args, model, device, train_loader, optimizer, lambda0, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        l1 = 0
        optimizer.zero_grad()  # set the initial gradient to be 0
        output, weightM = model(data)
        loss = nn.MSELoss()
        loss1 = loss(output[:, 0], target[:, 0])  # (output[:, 0] - target[:, 0]).pow(2).mean()
        # loss2 = (target[:, 0] * (output[:, 1] -target[:, 1])).pow(2).mean()/target[:, 0].mean()
        loss = loss1  # + lambda0 * loss2
        loss.backward()
        optimizer.step()  # update parameters
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return loss


def test(model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        outputm = torch.zeros(1, dtype=torch.float).to(device)
        targetm = torch.zeros(1, dtype=torch.float).to(device)
        for data, target in test_loader:
            data, target = data.to(device), target.float().to(device)
            output, weightM = model(data)
            outputm = torch.cat((outputm, output[:, 0].float()))
            targetm = torch.cat((targetm, target[:, 0]))
        loss = nn.MSELoss()
        loss1 = loss(outputm[1:], targetm[1:])
        loss = loss1
        outputm = outputm[1:].cpu().numpy()
        targetm = targetm[1:].cpu().numpy()
        outputmb = (outputm > 0.5).astype(float)
        acc = accuracy_score(targetm, outputmb)
        recall = recall_score(targetm, outputmb, average='binary')  # recall = TP / (TP + FN), find completely
        prec = precision_score(targetm, outputmb, average='binary')  # precision = TP / (TP + FP), find accurately
        spec = specificity_score(targetm, outputmb, average='binary')
        f1 = f1_score(targetm, outputmb, average='binary')  # f1 score = 2 * precision * recall / (precision + recall)
        aupc = average_precision_score(targetm, outputm)
        auc = roc_auc_score(targetm, outputm)
    result = {'acc': acc,
              'recall': recall,
              'prec': prec,
              'spec': spec,
              'f1': f1,
              'aupc': aupc,
              'auc': auc,
              'loss': loss.cpu().detach().numpy()}
    # print(pd.DataFrame(result, index=[0]))
    return result


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=10, metavar='N',
                        help='Learning rate step size (how many epochs) (default: 10)')
    parser.add_argument('--preprocess', action='store_true', default=True,
                        help='preprocess continuous eeg to segments (default: True)')
    parser.add_argument('--nX', type=int, default=250, metavar='N',
                        help='time length of X (default: 250 for TUH data)')
    parser.add_argument('--nZ', type=int, default=125, metavar='N',
                        help='left/half time length of Z (default: 125 for TUH data)')
    parser.add_argument('--ksizeX', type=int, default=2, metavar='N',
                        help='kernel size in CNN of X (default: 2)')
    parser.add_argument('--ksizeZ', type=int, default=2, metavar='N',
                        help='kernel size in CNN of Z (default: 2)')
    parser.add_argument('--scale', type=float, default=0.1, metavar='M',
                        help='scaling number for the second term of the weighted combination (default: 0.1)')
    parser.add_argument('--NCA', type=int, default=8, metavar='N',
                        help='number of neurons in the first layer to learn alpha (default: 8)')
    parser.add_argument('--NCG', type=int, default=32, metavar='N',
                        help='number of neurons in the first layer to learn g (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training (NVDA GPU: False; CPU: True)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (default: False)')
    parser.add_argument('--seed', type=int, default=8, metavar='S',
                        help='cuda random seed (default: 1)')
    parser.add_argument('--np-seed', type=int, default=2024, metavar='S',
                        help='numpy random seed to split training and testing samples (default: 2024)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--log-print-save', type=int, default=5, metavar='N',
                        help='how many batches to wait before saving the current model (default: 5)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For saving the current Model (default: True)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For load the existing Model (default: False)')

    args = parser.parse_args()

    # use cuda if detected
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory and set seed
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)

    print("=======CUDA AVAILABLE========")
    print(torch.cuda.is_available())
    print("===================")
    print("=======DEVICE TYPE========")
    print(device)
    print("===================")
    print("::")
    print("-----------PARAMETERS-----------")
    print(args)
    print("--------------------------------")

    # change working dir to the project path
    # please change the working dir to where you save this folder in your local device
    work_dir = '/Users/shirleywei/Dropbox/Projects/NestedDeepLearningModel/NDL_BrainSignal_RepCode_Mac'
    os.chdir(work_dir)

    # path to read raw data
    # !!! remember to modify to your own directory where TUH data are saved
    raw_data_dir = work_dir + '/data/tuev/edf/'

    # path to write pre-processed segmented data
    processed_data_dir = work_dir + '/data/segment/'
    os.makedirs(processed_data_dir, exist_ok=True)

    # path to save model and output
    model_name = 'tuh_' + '_' + str(args.lr) + '_' + str(args.gamma) + '_' + str(args.step_size) + '.pt'
    model_dir = work_dir + '/model/' + model_name

    # montage
    channel_anode = [
        'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF',
        'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
        'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
        'EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
        'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF'
    ]
    channel_cathode = [
        'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF',
        'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF',
        'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
        'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF',
        'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
    ]

    # preprocess continuous eeg to segments
    if args.preprocess:
        print('Preprocessing EEG data...')
        continuous2segmentation(read_dir=raw_data_dir, write_dir=processed_data_dir, data_type='train',
                                channel_anode=channel_anode, channel_cathode=channel_cathode, nX=args.nX, nZ=args.nZ)
        continuous2segmentation(read_dir=raw_data_dir, write_dir=processed_data_dir, data_type='eval',
                                channel_anode=channel_anode, channel_cathode=channel_cathode, nX=args.nX, nZ=args.nZ)
        print('Done'), print('\n')

    # load segment data dir
    path = processed_data_dir
    filenames = os.listdir(path)
    filenames = [f for f in filenames if not f.startswith('.')]

    # set numpy random seed, and split train and test set
    np.random.seed(args.np_seed)
    n = len(filenames)

    ixtrain = np.random.choice(np.arange(n), int(n * 0.8), replace=False)  # randomly select as the train
    ixtest = np.setdiff1d(np.arange(n), ixtrain)  # the rest as the test
    trainid = []
    for i in ixtrain:
        trainid.append(filenames[i])
    testid = []
    for i in ixtest:
        testid.append(filenames[i])
    partition = {'train': trainid, 'test': testid}

    # data loader parameters
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    para = {
        'T': args.nX,
        'p': args.nZ * 2
    }

    # load data
    training_set = Dataset(partition['train'], path, para)
    train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs)
    test_set = Dataset(partition['test'], path, para)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    # network parameters
    NC1A = args.NCA
    NC2A = args.NCA * 2
    NC3A = args.NCA
    NC4 = 1
    NC1G = args.NCG
    NC2G = args.NCG * 2
    NC3G = args.NCG

    # initialize the model
    if args.load_model:  # maybe set False if not trained before
        model = Net(NC1A, NC2A, NC3A, NC1G, NC2G, NC3G, args.nX, args.nZ, args.ksizeX, args.ksizeZ, args.scale).to(
            device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        if torch.cuda.is_available():
            ckpoint = torch.load(model_dir)
        else:
            ckpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        model.load_state_dict(ckpoint['model_state_dict'])
        optimizer.load_state_dict(ckpoint['optimizer_state_dict'])
        epoch0 = ckpoint['epoch']
        loss = ckpoint['loss']
    else:
        model = Net(NC1A, NC2A, NC3A, NC1G, NC2G, NC3G, args.nX, args.nZ, args.ksizeX, args.ksizeZ, args.scale).to(
            device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        epoch0 = 0

    # scheduler
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # train the model
    for epoch in range(epoch0 + 1, args.epochs + epoch0 + 1):
        loss = train(args, model, device, train_loader, optimizer, 0, epoch)
        result = test(model, device, test_loader, epoch)
        scheduler.step()

        # save model
        if args.save_model and (epoch % args.log_print_save == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_dir)
            print(pd.DataFrame(result, index=[0]))


if __name__ == '__main__':
    main()
