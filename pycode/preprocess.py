import mne
import os
import os.path as osp
import numpy as np
from scipy import signal
import scipy.io


# preprocess functions

# read raw eeg data in edf format

def read_edf_data(data_path):
    # read raw edf data
    rawdata = mne.io.read_raw_edf(data_path, verbose=False)

    # copy the raw data
    data = rawdata.copy().get_data()

    # get sampling frequency
    sfreq = rawdata.info['sfreq']

    # get channel names
    chnames = rawdata.ch_names

    return data, sfreq, chnames


# read raw eeg data in m00 format

def read_m00_data(data_path):
    # read raw m00 eeg data
    f = open(data_path, "r")
    txt = f.readlines()[0]
    data = np.loadtxt(data_path, skiprows=2)

    sfreq = 1000 / float(txt.split(' ')[3].split('=')[1])
    data = - data[:, :19].transpose()

    return data, sfreq


# make tcp 22 montages

def make_tcp_montage(data, chnames, channel_anode, channel_cathode):
    # create tcp 22 montage
    order_anode = [chnames.index(ch) for ch in channel_anode]
    order_cathode = [chnames.index(ch) for ch in channel_cathode]
    data = data[order_anode] - data[order_cathode]
    data = -data

    return data


# make average montage

def make_average_montage(data):
    data = data - data.mean(0)[np.newaxis, :]

    return data


# preprocess the raw data: resample, filter, and detrend

# downrate: the resampled sampling frequency, 250 Hz for TUH eeg data

# l_freq, h_freq: filtered low frequency (1 Hz) and high frequency (45 Hz - 70 Hz)

# notch_freq: 60 for US eeg data, 50 for China eeg data

def preprocessing(data, sfreq, downrate=250, l_freq=1, h_freq=70, notch_freq=60):
    # resample the data
    data = mne.filter.resample(data, up=1, down=sfreq / downrate, verbose=False)

    # filter the data
    data = mne.filter.filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False) * 1e7
    data = mne.filter.notch_filter(data, Fs=downrate, freqs=notch_freq, verbose=False)

    # detrend the data
    data = mne.epochs.detrend(data)

    return data


# read annotation records of each eeg data

def read_target_file(rec_path):
    # load record file that stores event time
    rec_txt = np.loadtxt(rec_path, delimiter=',')

    # convert 6 classes of events into abnormal and normal target
    cond_label = rec_txt[:, 3] <= 3
    cond_label = cond_label.astype('int')

    # combine the original table with the binary target, with columns [ch_loc, t_start, t_end, 1-6_types, 0/1_lab]
    rec_txt = np.c_[rec_txt, cond_label]

    # round the time to .1 seconds, then sort by time column, ascending
    # not necessary to remove 90% overlapping after round time to 0.1, largest overlapping = 0.9 / 1.1 < 90%
    rec_txt = np.round(rec_txt, 1)
    rec_txt = rec_txt[rec_txt[:, 1].argsort()]

    # select two columns of time, then only keep unique time pairs
    rec_time = rec_txt[:, 1:3]
    rec_time = np.unique(rec_time, axis=0)

    return rec_time, rec_txt


# to be called in train.py file

# the packaged function for the whole preprocessing procedure

# convert the raw continuous eeg data into preprocessed segments (X, Z) with a corresponding binary target Y

def continuous2segmentation(read_dir, write_dir, data_type: str,
                            channel_anode, channel_cathode, nX: int, nZ: int,
                            downrate=250, l_freq=1, h_freq=70, notch_freq=60):
    # counter
    if data_type == 'train':
        sample_id = 0
    elif data_type == 'eval':
        sample_id = 10000
    else:
        raise ValueError('data_type must be "train" or "eval"')

    # list subject IDs
    data_path = osp.join(read_dir, data_type)
    subject_id = [f for f in os.listdir(data_path) if not f.startswith('.')]
    subject_id.sort()

    # read experiments under each subject
    for subject in subject_id:  # subject = '00000021'
        subject_path = osp.join(data_path, subject)
        subject_experiment_id = [ex[:-4] for ex in os.listdir(subject_path) if 'edf' in ex]  # ['00000021_00000001']
        spike_type_per_patient = np.array([])
        subject_experiment_id.sort()

        # read raw edf data under each experiment of the subject
        for sub_experiment in subject_experiment_id:  # sub_experiment = '00000021_00000001'
            # experiment = sub_experiment[9:]  # '00000001'
            edf_filepath = osp.join(subject_path, sub_experiment + '.edf')
            rec_filepath = osp.join(subject_path, sub_experiment + '.rec')

            # read raw edf data
            try:
                rawdata, sfreq, chnames = read_edf_data(edf_filepath)

            # raise error if fail to load the data, then continue to read the next one
            except Exception as error:
                print("An error occurred:", error)
                print(edf_filepath)
                continue

            # make montage
            data = make_tcp_montage(rawdata, chnames, channel_anode, channel_cathode)

            # preprocess data
            subeeg = preprocessing(data, sfreq, downrate, l_freq, h_freq, notch_freq)

            # read event time
            rec_time, rec_txt = read_target_file(rec_filepath)

            # read record time pairs
            for rec in range(rec_time.shape[0]):

                # start time and end time of signals X
                t_start = float(rec_time[rec, 0])
                t_end = float(rec_time[rec, 1])

                # convert time (seconds) to time index
                nt_start = round(t_start * nX)
                nt_end = round(t_end * nX)

                # include reference signals Z
                t0_start = nt_start - nZ
                t0_end = nt_end + nZ

                # drop index over range
                if t0_start >= 0 and t0_end <= subeeg.shape[1]:
                    # segmentation
                    eeg = subeeg[:, t0_start: t0_end]

                    # detrend eeg
                    eeg_dtr = signal.detrend(eeg, axis=-1, type='linear')

                    # find all events during this period
                    cond_channel = np.where(
                        (rec_txt[:, 1] == t_start) & (rec_txt[:, 2] == t_end)
                    )
                    info = rec_txt[cond_channel, :][0]  # select info during the period
                    spike_type = np.unique(info[:, -2])  # 1-6 unique types in the 1s epoch
                    spike_type_per_patient = np.concatenate((spike_type_per_patient, spike_type))

                    # binary target, 1 if the segment contains at least one abnormal event
                    target = np.array([max(info[:, -1])])

                    # count sample ID
                    sample_id += 1

                    # info table, with columns [ch_loc, 1-6 types] for this interval
                    info = info[:, [0, -2]]

                    # start and end time
                    time_start_end = np.array([t_start, t_end])

                    # subject and experiment info
                    filename = np.array([sub_experiment])

                    # save segmented data
                    np.savez(osp.join(write_dir, str(sample_id) + '.npz'),
                             eeg=eeg_dtr, target=target, info=info,
                             time=time_start_end, filename=filename)
