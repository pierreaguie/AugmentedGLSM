# Augmentation Invariant Generative Language Speech Modeling (GSLM)

## TODO

- Reproduce the encoder-quantizer pipeline
- Data processing for LibriSpeech-100h
- Implement the different augmentations



## Usage

To run this repository, you will need Python 3.9, and to install [textlesslib](https://github.com/facebookresearch/textlesslib) and [fairseq](https://github.com/facebookresearch/fairseq).


```
# Create a dedicated environment
python3.9 -m venv GLSMEnv/ 
source GLSMEnv/bin/activate

# Install textlesslib
git clone https://github.com/facebookresearch/textlesslib.git
cd textlesslib
pip install -e .

# Install fairseq
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8
```

To install the [LibriSpeech](https://www.openslr.org/12) datasets (training, 6.3G; test, 346M) directly into the `data` directory, run the following command:
```
mkdir data
cd data
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf train-clean-100.tar.gz
tar -xzvf test-clean.tar.gz
```


## To load textlesslib's SpeechEncoder:

Modify line 304 of checkpoints_utils.py : in torch.load, add weights_only=False
