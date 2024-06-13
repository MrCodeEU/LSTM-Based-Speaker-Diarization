from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    def __init__(self, log_mel_spectrograms, labels):
        self.log_mel_spectrograms = log_mel_spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.log_mel_spectrograms)

    def __getitem__(self, idx):
        return self.log_mel_spectrograms[idx], self.labels[idx]
