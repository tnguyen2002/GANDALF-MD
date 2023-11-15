import os
import copy
import torch
import random
import librosa
import torchaudio
import numpy as np
from glob import glob
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize

from torch.utils.data import Dataset
from nlpaug.augmenter.audio import AudioAugmenter
from src.datasets.librispeech import WavformAugmentation, SpectrumAugmentation

from src.datasets.root_paths import DATA_ROOTS

AUDIOMNIST_MEAN = [-90.293]
AUDIOMNIST_STDEV = [11.799]
AUDIOMNIST_TRAIN_SPK = [28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                        8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]
AUDIOMNIST_VAL_SPK = [12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]
AUDIOMNIST_TEST_SPK = [26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]

AUDIOMNIST_SAMPLE_RATE = 48000

'''
There are three classes of background noises, each with different statistical characteristics, that can be added to each AudioMNIST audio file as the suppressing feature.
Only one audio file from each class is used and samples of each audio file are taken at random timestamps. 
More details can be found here: https://personal.utdallas.edu/~nxk019000/VAD-dataset/EMBC2016-Saki-DatasetReadMe.pdf
'''
NOISES_PATHS = ["./noisedata/BabbleNoise/Babble1.wav", "./noisedata/DrivingcarNoise/Traffic10.wav", "./noisedata/MachineryNoise/Machinery1.wav"]
NOISES = [None, None, None]
for i, noise in enumerate(NOISES_PATHS):
    NOISES[i], _ = librosa.load(noise, sr=AUDIOMNIST_SAMPLE_RATE)

class AudioMNIST(Dataset):

    def __init__(
            self,
            root=DATA_ROOTS['audio_mnist'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=True,
            max_length=50000,
            input_size=224,
            normalize_mean=AUDIOMNIST_MEAN,
            normalize_stdev=AUDIOMNIST_STDEV,
            noise_volume=0,
            alternate_label=False,
            feat_dropout_prob=0
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        if train:
            speakers = AUDIOMNIST_TRAIN_SPK + AUDIOMNIST_VAL_SPK
        else:
            speakers = AUDIOMNIST_TEST_SPK
        wav_paths = []
        for spk in speakers:
            spk_paths = glob(os.path.join(root, "{:02d}".format(spk), '*.wav'))
            wav_paths.extend(spk_paths)
        self.wav_paths = wav_paths
        self.num_labels = 10
        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.max_length = max_length
        self.train = train
        self.input_size = input_size
        self.FILTER_SIZE = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev
        self.noise_volume = noise_volume
        self.alternate_label = alternate_label
        all_speaker_ids = self.get_speaker_ids()
        unique_speaker_ids = sorted(list(set(all_speaker_ids)))
        num_unique_speakers = len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(num_unique_speakers)))
        self.all_speaker_ids = np.array([self.speaker_id_map[sid] for sid in all_speaker_ids])

        if self.alternate_label:
            self.num_labels = 3
        
        self.feat_dropout_prob = feat_dropout_prob
    
    def get_speaker_ids(self):
        speaker_ids = []
        for wav_path in self.wav_paths:
            speaker_id, _, _ = wav_path.rstrip(".wav").split("/")[-1].split("_")
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)


    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label, _, _ = wav_path.rstrip(".wav").split("/")[-1].split("_")

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        noise_idx = random.randint(0, 2)
        noise = NOISES[noise_idx]

        max_peak = np.max(np.abs(noise))
        ratio = self.noise_volume / max_peak
        noise = noise * ratio
        
        if len(noise) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded_noise = (noise[:self.max_length] if flip else 
                      noise[-self.max_length:])
        else:
            padded_noise = np.zeros(self.max_length)
            padded_noise[:len(noise)] = noise  # pad w/ silence
        
        if self.wavform_transforms:
            # pad to 150k frames
            if len(wavform) > self.max_length:
                # randomly pick which side to chop off (fix if validation)
                flip = (bool(random.getrandbits(1)) if self.train else True)
                padded = (wavform[:self.max_length] if flip else 
                        wavform[-self.max_length:])
            else:
                padded = np.zeros(self.max_length)
                padded[:len(wavform)] = wavform  # pad w/ silence
            padded = padded + padded_noise

            if self.wavform_transforms:
                transforms = WavformAugmentation(sample_rate)
                padded = transforms(padded)
        else:        
            # pad to 150k frames
            if len(wavform) > self.max_length:
                # randomly pick which side to chop off (fix if validation)
                flip = (bool(random.getrandbits(1)) if self.train else True)
                padded = (wavform[:self.max_length] if flip else 
                        wavform[-self.max_length:])
            else:
                padded = np.zeros(self.max_length)
                padded[:len(wavform)] = wavform  # pad w/ silence
            padded = padded + padded_noise
        
        hop_length_dict = {64: 788}
        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=hop_length_dict[self.input_size],
            n_mels=self.input_size,
        )

        if self.spectral_transforms:  # apply time and frequency masks
            transforms = SpectrumAugmentation()
            spectrum = transforms(spectrum)

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        if self.spectral_transforms:  # apply noise on spectral
            noise_stdev = 0.25 * self.normalize_stdev[0]
            noise = torch.randn_like(spectrum) * noise_stdev
            spectrum = spectrum + noise

        normalize = Normalize(self.normalize_mean, self.normalize_stdev)
        spectrum = normalize(spectrum)
        
        if self.alternate_label:
            label = noise_idx

        return index, spectrum, int(label)

    def __len__(self):
        return len(self.wav_paths)

class AudioMNISTTwoViews(AudioMNIST):

    def __getitem__(self, index):
        index, spectrum1, label = super().__getitem__(index)

        p = random.random()
        _, spectrum2, _ = super().__getitem__(index) if p < self.feat_dropout_prob else (index, copy.deepcopy(spectrum1), label)

        return index, spectrum1, spectrum2, label
