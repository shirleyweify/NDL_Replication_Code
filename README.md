# Nested Deep Learning (NDL) Framework - Read Me

```
Title: Nested Deep Learning Model Towards a Foundation Model for Brain Signal Data
Keywords: epilepsy, Electroencephalography, spike detection, deep learning, neuroimaging
ArXiv link: https://arxiv.org/abs/2410.03191
```

If you would like to use the codes, please cite:

**Wei, Fangyi, Jiajie Mo, Kai Zhang, Haipeng Shen, Srikantan Nagarajan, and Fei Jiang. "Nested deep learning model towards a foundation model for brain signal data." arXiv preprint arXiv:2410.03191 (2024).**

Please send your formal request to [fwei@connect.hku.hk](mailto:fwei@connect.hku.hk) if any.

The work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/).


---

## Data

We apply NDL to a public dataset (**TUH data**) and two private datasets (**UCSF MEG data** and **BTH EEG data**).

**TUH Data**

To replicate the results from the open data (TUH data) in the paper, please download the raw data from the following
website link:

- **TUH Data**: public EEG data, download
  from [The TUH EEG Events Corpus (TUEV: v2.0.0)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

```
1. To download the dataset, please follow the instructions on the website and request permissions from the data provider.
2. Spare more than 20 G storage for this dataset.
```

Please download and save the TUH data into the directory `data/*`, where `*` should be the name of your folder
downloaded from the website.
For example, I name it as `data/tuev/`.
Then you should have `data/tuev/edf/train/`, `data/tuev/edf/eval/` and a readme txt after downloading the TUH data.
Please download the version 2.0.0 to replicate the exact results from the paper.

**UCSF MEG Data** and **BTH EEG Data**

We cannot provide the private datasets as they are sensitive and credential clinical data.
But our model supports training and testing using various formats of EEG or MEG data.
Please try NDL with your own datasets let me know if you have any problem.

---

## Replication Codes

The replication codes for the Nested Deep Learning (NDL) framework include preprocessing, training, testing, and the NDL
neural networks.

All the codes are under the `pycode/` folder, where
`ndlnet.py` is the neural network function,
`preprocess.py` contains a packaged function that preprocess raw data to segmented data,
`train.py` is to train models using `ndlnet.py`,
and `test.py` is to test fine-tuned models on raw data and generate spike annotations.

### Preprocess

Before training, we need to preprocess continuous raw EEG data into segmented multi-channel signals.

`preprocess.py` stores functions that preprocess continuous raw edf data into segments.
By setting `--preprocess True` in `train.py`,
the preprocess function `continuous2segmentation()` will be automatically called from `preprocess.py`.

The `continuous2segmentation()` function automatically processes the raw edf data saved under `data/tuev/edf/train/` and
`data/tuev/edf/test/` into segments and saves the segments under `data/segment/`.

Only set `--preprocess True` for the first time when you run `train.py`.
After that, please set `--preprocess False` for your future training procedures.

### Training

Run `train.py` with the fine-tuned parameters to replicate our results in the paper.
Please change the `work_dir` to your own repository location.
Set `--preprocess True` when you run `train.py` for the first time.

We provide the fine-tuned model `model/tuh_fine_tuned.pt` trained with TUH data.
Skip this step and jump to `test.py` if you would like to test with the model we provide rather than the model trained
by your own.

### Testing

To test the fine-tuned model `model/tuh_fine_tuned.pt` on raw edf data, run `test.py`.

`test.py` aims to read the raw data (in `.edf` formats) under `data/test/` and output annotation json files to
`data/annotation/`.
You may write your own functions to read the raw data in other formats such as `.m00` and so on.

The model `model/tuh_fine_tuned.pt` is trained using the open-source TUH data.
To require the model trained using the private UCSF MEG data, please send your formal request
to [fwei@connect.hku.hk](mailto:fwei@connect.hku.hk).

---
