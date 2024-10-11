# load audio file
import os
import random
from audiomentations import AddBackgroundNoise
import sounddevice as sd
from utils import add_noise_augmentation
import librosa

from config import Hyperparameters

audio_path = "data_vox/train/id10001/1zcIwhmdeo4/00001.wav"
audio, sr = librosa.load(audio_path, sr=None)

hp = Hyperparameters()

# add background noise
if hp.noise:
    # Load noise data and create the augment object once
    noise_augment = AddBackgroundNoise(
        sounds_path=hp.noise_dir,
        p=1
    )

    audio = add_noise_augmentation(audio, sr, noise_augment)

# play audio without IPython.display
sd.play(audio, sr)
sd.wait()
