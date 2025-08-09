from __future__ import print_function
from ndlnet import *
from preprocess import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io
from scipy import signal
from scipy.spatial import distance_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import os
import mne
import pickle
import pandas as pd
import os.path as osp
from sklearn.cluster import DBSCAN
import json

torch.cuda.empty_cache()


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, segment, data):
        'Initialization'
        self.list_IDs = list_IDs
        self.segment = segment
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        time = self.segment[ID]
        X = self.data[:, time]
        res = X[:, :] - X[:, :].mean(0)[np.newaxis, :]
        res = X[:, :] - X[:, :].mean(1)[:, np.newaxis]
        res = torch.from_numpy(res).unsqueeze(0).float()
        res = res / res.std()
        return res


def getCloseIndex1(megpos, eegpos, megdata):
    dxy = distance_matrix(megpos, eegpos)
    dxymin = np.argmin(dxy, 1)
    index = []
    megres = np.array(pd.DataFrame(megdata).groupby(dxymin).mean(0))
    return megres


def getNeighb(megdata, ch_names, ctfDict):
    keys = list(ctfDict.keys())
    submeg = np.zeros((len(keys), megdata.shape[1]))
    for i in range(len(ctfDict)):
        ix = np.isin(ch_names, ctfDict[keys[i]])
        submeg[i, :] = np.nanmean(megdata[ix,], 0)
    return submeg


def getCloseIndex(megpos, eegpos, megdata):
    dxy = distance_matrix(megpos, eegpos)
    dxymin = np.argmin(dxy, 0)
    megres = megdata[dxymin, :]
    return megres


def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        outputm = torch.zeros(1, dtype=torch.float).to(device)
        targetm = torch.zeros(1, dtype=torch.float).to(device)
        weightm = np.array([]).reshape(0, 22)
        for data in test_loader:
            data = data.float().to(device)
            output, weight = model(data)
            outputm = torch.cat((outputm, output[:, 0].float()))
            # targetm = torch.cat((targetm, output[:, 1].float()))
            weight = weight.cpu().numpy().squeeze().sum(2)
            weightm = np.concatenate((weightm, weight), axis=0)
        outputm = outputm[1:].cpu().numpy()
        # targetm = targetm[1:].cpu().numpy() * 0.25
        return outputm, weightm  # targetm,


# postprocess functions

def skew(subeeg):
    n = len(subeeg)
    denom = subeeg.sum()
    x = np.arange(0, n)
    E1 = np.sum(x * subeeg / denom)
    E2 = ((np.sum(x ** 2 * subeeg / denom) - E1 ** 2) ** (1 / 2)) + 1e-6
    E3 = np.sum(((x - E1) / E2) ** 3 * subeeg / denom)
    return (E3)


# find peaks for eeg
# nt: time length of each segment (= nX + nZ * 2)

def find_peak_eeg(eegdata, downrate: int, nt):
    # initialize an arrary to record the location of the peaks
    last_spike_peak = np.array([1])

    # find peaks for each channel
    for itr in range(eegdata.shape[0]):
        subeeg = np.abs(eegdata[itr, :])
        peaks = scipy.signal.find_peaks(subeeg, distance=downrate)
        spike_peak = peaks[0]  # np.intersect1d(spikes,peaks[0])
        widths, height, left, right = scipy.signal.peak_widths(subeeg, peaks[0])
        left = np.round(left).astype(int)
        right = np.round(right).astype(int)
        skewm = np.zeros(len(left))
        for i in range(len(left)):
            skewm[i] = skew(subeeg[left[i]:right[i]])
        skewm = np.abs(skewm)
        width = widths / downrate
        spike_peak = spike_peak[
            (width <= 0.20) & (width >= 0.04) & (height > np.mean(height) / 2) & (skewm > np.median(skewm))]
        last_spike_peak = np.union1d(last_spike_peak, spike_peak)

    # remove peaks out of the range
    last_spike_peak = list(last_spike_peak[(last_spike_peak > nt / 2) & (last_spike_peak <= eegdata.shape[1] - nt / 2)])
    last_spike_peak = np.array(last_spike_peak)
    return last_spike_peak


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=10, metavar='N',
                        help='Learning rate step size (how many epochs) (default: 10)')
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
    parser.add_argument('--thres', type=float, default=0.95, metavar='M',
                        help='Threshold for prediction selection (default: 0.995)')
    parser.add_argument('--downrate', type=int, default=250, metavar='N',
                        help='downrate sampling frequency (default: 250 for TUH data; 256 for UCSF MEG or BTH EEG data)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)

    # change working dir to the project path
    # please change the working dir to where you save this folder in your local device
    work_dir = '/Users/shirleywei/Dropbox/Projects/NestedDeepLearningModel/NDL_BrainSignal_RepCode_Mac'
    os.chdir(work_dir)

    # path to read raw data
    # ---------- #
    # IMPORTANT
    # note: copy paste the raw data you would like to test into the `test` folder
    # otherwise, no data will be available under the `test` folder
    # ---------- #
    raw_data_dir = work_dir + '/data/test/'
    os.makedirs(raw_data_dir, exist_ok=True)

    # path to write pre-processed segmented data
    annotated_data_dir = work_dir + '/data/annotation/'
    os.makedirs(annotated_data_dir, exist_ok=True)

    # path to save model and output
    model_name = 'tuh_fine_tuned.pt'
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
    tcp_montage_channel = [
        'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
    ]

    # load raw eeg or meg data
    # list subject IDs
    filename = [f for f in os.listdir(raw_data_dir) if not f.startswith('.')]
    filename.sort()

    for file in filename:
        file_path = raw_data_dir + file

        # read raw edf data
        try:
            rawdata, sfreq, chnames = read_edf_data(file_path)

        # raise error if fail to load the data, then continue to read the next one
        except Exception as error:
            print("An error occurred:", error)
            print(file_path)
            continue

        # make montage
        data = make_tcp_montage(rawdata, chnames, channel_anode, channel_cathode)

        # preprocess data
        subeeg = preprocessing(data, sfreq, downrate=args.downrate, l_freq=1, h_freq=70, notch_freq=60)

        # find peaks
        nt = round(args.nX + args.nZ * 2)
        last_spike_peak = find_peak_eeg(eegdata=subeeg, downrate=args.downrate, nt=nt)

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

        # load data according peak locations
        testid = []
        Dict = {}
        for i in last_spike_peak:
            key = 'id-' + str(i)
            Dict[key] = range(int(i - nt / 2), int(i + nt / 2))
            testid.append('id-' + str(i))
        test_set = Dataset(testid, Dict, subeeg)
        test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

        # network parameters
        NC1A = args.NCA
        NC2A = args.NCA * 2
        NC3A = args.NCA
        NC4 = 1
        NC1G = args.NCG
        NC2G = args.NCG * 2
        NC3G = args.NCG

        # load the model for testing
        model = Net(NC1A, NC2A, NC3A, NC1G, NC2G, NC3G, args.nX, args.nZ, args.ksizeX, args.ksizeZ, args.scale).to(
            device)
        if torch.cuda.is_available():
            ckpoint = torch.load(model_dir)
        else:
            ckpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        model.load_state_dict(ckpoint['model_state_dict'])

        # test the model
        detection, weight = test(model, device, test_loader)

        # remove duplicates
        # set parameters for selection
        ndev = 1.5
        topn = 1

        # select locations where the predicted probability larger than the threshold
        # time length for each segment to be half of (X, Z), 1s for TUH and 0.25s for MEG
        sample_idx = np.where(detection > args.thres)[0]
        bf = last_spike_peak[sample_idx]  # middle point of the peaks, also the start rec index, not true start
        af = last_spike_peak[sample_idx] + round(nt / 2)  # end rec index, not true end index
        subdet = detection[sample_idx]

        # DBSCAN to remove duplicates
        db = DBSCAN(eps=round(nt / 4), min_samples=1).fit(bf.reshape(len(bf), 1))
        labels = db.labels_
        sublabel = labels[labels > -1]
        bincount = np.bincount(sublabel)
        ulab = np.unique(sublabel)
        itrlabel = ulab[(bincount >= 1)]

        sbf = np.zeros(len(itrlabel))
        saf = np.zeros(len(itrlabel))
        value = np.zeros(len(itrlabel))
        dis = np.zeros(len(itrlabel))
        sidx = np.zeros(len(itrlabel))
        for itr in range(len(itrlabel)):
            litr = itrlabel[itr]
            subbf = bf[labels == litr]
            subaf = af[labels == litr]
            subidx = sample_idx[labels == litr]
            aix = np.argmax(subdet[labels == litr])
            value[itr] = np.max(subdet[labels == litr])
            sbf[itr] = subbf[aix]
            saf[itr] = sbf[itr] + (nt / 2)
            sidx[itr] = subidx[aix]
            dis[itr] = len(subbf)

        # sbf is the middle point, saf is (+1 *dsHZ for TUH method, +0.25dsHZ for MEG method)
        # end is fixed so we do not need to record
        # value is the probability of being spike
        events = np.vstack((sbf, np.zeros(len(sbf)), np.ones(len(sbf)))).transpose()  # index of each event

        # highlight the top channels selected by weights
        n_blinks = events.shape[0]

        # sort the top n channels by descending weights
        weight_sort = (np.argsort(-weight, 1))[:, :].reshape([-1, weight.shape[1]])
        topn_sort = weight_sort[:, :topn].reshape([-1, topn])

        # record the top channels
        ch_names = []
        tcp_montage_channel = np.array(tcp_montage_channel)
        for s in sidx.astype(int):
            wt_ar = weight[s, :]
            wt_thres = wt_ar.mean() * ndev
            select_ch = tcp_montage_channel[np.where(wt_ar > wt_thres)[0]].tolist()
            if len(select_ch) < topn:
                topn_channel_index = topn_sort[s, :].astype(int)
                select_ch = tcp_montage_channel[topn_channel_index]
            select_ch = list(select_ch)
            ch_names.append(select_ch)



        # # To visualize the annotations using mne
        # onset = (events[:, 0] - nt / 4) / args.downrate  # start time of each event
        # duration = np.repeat((nt / args.downrate) * 0.5, n_blinks)  # 1s for TUH, 0.25s for MEG
        # description = ['event'] * n_blinks
        # info = mne.create_info(list(tcp_montage_channel), sfreq=args.downrate)
        # rawdata = mne.io.RawArray(subeeg, info)
        # annotations = mne.Annotations(onset=onset, duration=duration, description=description, ch_names=ch_names)
        # rawdata.set_annotations(annotations)
        # rawdata.plot(block=True)

        # # To save the raw data with annotations using mne
        # rawdata.save(osp.join(annotated_data_dir, file[:-4] + '_eeg.fif'), overwrite=True)


        # save annotations to json files
        # convert format from numpy array to list
        anno_time = list(events[:, 0] / args.downrate)  # 1D, annotated time index
        anno_prob = list(value)  # 1D, corresponding segment probability
        anno_ch = ch_names  # 2D, annotated index w/ top channels
        anno = {
            "time": anno_time,
            "probability": anno_prob,
            "channels": anno_ch
        }

        json_filename = annotated_data_dir + 'anno_' + file[:-4] + '.json'
        with open(json_filename, 'w') as f:
            json.dump(anno, f, indent=4)

        # print finish text
        print('File ' + file + ' annotated.')



if __name__ == '__main__':
    main()
