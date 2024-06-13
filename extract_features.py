import librosa.feature
import numpy as np
import torch


def extract_features(segment, sr, n_mels=40, n_fft=256, hop_length=80):
    # Ensure segment is a numpy array
    audio_segment = segment.numpy() if isinstance(segment, torch.Tensor) else segment

    # Extract log mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     hop_length=hop_length, power=1)
    db_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return db_spectrogram.T
