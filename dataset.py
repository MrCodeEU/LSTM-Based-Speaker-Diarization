import random
from torch.utils.data import BatchSampler, Dataset


class SpeakerDataset(Dataset):
    def __init__(self, log_mel_spectrograms, labels):
        self.data = []
        speaker_to_idx = {}
        for idx, label in enumerate(labels):
            if label not in speaker_to_idx:
                speaker_to_idx[label] = []
            speaker_to_idx[label].append(idx)

        for speaker, indices in speaker_to_idx.items():
            self.data.extend([(log_mel_spectrograms[idx], speaker) for idx in indices])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultipleSpeakersBatchSampler(BatchSampler):
    def __init__(self, speaker_to_indices, speakers_per_batch, samples_per_speaker):
        self.speaker_to_indices = speaker_to_indices
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.batch_size = speakers_per_batch * samples_per_speaker

    def __iter__(self):
        indices_per_speaker = {speaker: list(indices) for speaker, indices in self.speaker_to_indices.items()}
        num_batches = len(self)

        for _ in range(num_batches):
            speakers = random.sample(list(indices_per_speaker.keys()), self.speakers_per_batch)
            batch = []

            for speaker in speakers:
                if len(indices_per_speaker[speaker]) >= self.samples_per_speaker:
                    batch.extend(random.sample(indices_per_speaker[speaker], self.samples_per_speaker))
                else:
                    batch.extend(indices_per_speaker[speaker])
                    indices_per_speaker[speaker] = []

            yield batch

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.speaker_to_indices.values())
        num_batches = total_samples // self.batch_size
        return num_batches
