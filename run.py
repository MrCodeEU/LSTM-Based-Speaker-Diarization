import os
import random
import librosa
import numpy as np
import soundfile as sf
import torch
#from datasets import load_dataset
from pyannote.core import Annotation, Segment
from clustering import spectral_clustering, offline_kmeans
from config import Hyperparameters
from eval import eval_diarization, annotation_to_rttm
from extract_features import extract_features
from model import LSTMModel
from preprocessing import apply_vad, segment_audio
from utils import create_mixed_audio, load_audio_and_rttm, load_dataset_callhome
from visualize import visualize_diarization


def filter_short_segments(speaker_segments, threshold=0.5):
    """
    Filters out consecutive speaker segments with a total duration below the threshold.
    """
    if threshold == 0:
        return speaker_segments
    filtered_segments = []
    current_speaker = None
    current_duration = 0
    for start, end, speaker_id in speaker_segments:
        if speaker_id != current_speaker:  # New speaker or the start
            if current_duration >= threshold:  # Check duration of the previous speaker
                filtered_segments.extend(current_speaker_segments)  # Add segments if long enough
            current_speaker = speaker_id
            current_speaker_segments = [(start, end, speaker_id)]
            current_duration = end - start
        else:  # Same speaker, continue accumulating duration
            current_speaker_segments.append((start, end, speaker_id))
            current_duration += end - start

    # Handle the last speaker's segments
    if current_duration >= threshold:
        filtered_segments.extend(current_speaker_segments)

    return filtered_segments


def average_d_vectors(segment_d_vectors):
    """Averages a set of d-vectors for a single segment."""

    # L2 normalize d-vectors
    normalized_d_vectors = segment_d_vectors / np.linalg.norm(segment_d_vectors, axis=1, keepdims=True)

    # Average normalized d-vectors
    average_d_vector = np.mean(normalized_d_vectors, axis=0)

    return average_d_vector


def run_diarization(audio_path, model_path, segment_length=30, overlap=10, num_speakers=None,
                    target_segment_length=200, seed=None, sigma=6.5, percentile=60, callhome_index=0):
    hp = Hyperparameters()
    # Load the trained LSTM model
    input_size = hp.input_size
    hidden_size = hp.hidden_size
    num_layers = hp.num_layers
    projection_size = hp.projection_size
    model = LSTMModel(input_size, hidden_size, num_layers, projection_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    ground_truth_segments = None

    if hp.eval_dataset == "data_icsi":
        # Load the audio and extract log mel features
        audio, sr = librosa.load(audio_path, sr=None)
    elif hp.eval_dataset == "data_vox":
        # load n random speakers m audio files and splice them together to create a new audio file
        n = random.randint(2, 5)
        m = random.randint(2, 15)
        audio, sr, ground_truth_segments = create_mixed_audio(hp.eval_dataset, n, m, seed=seed)

        # save the audio to a file
        audio_path_save = "mixed_audio.wav"
        sf.write(audio_path_save, audio, sr)
    elif hp.eval_dataset == "data_conv":
        # load voxconverse audio and rttm file for ground truth
        audio, sr, ground_truth_segments = load_audio_and_rttm(audio_path)

        # print number of speakers in the ground truth
        print(f"Number of speakers in the ground truth: {len(ground_truth_segments.labels())}")
    else:
        # load callhome dataset from duckdb file
        audio, sr, ground_truth_segments = load_dataset_callhome(audio_path, callhome_index)
        # print number of speakers in the ground truth
        print(f"Number of speakers in the ground truth: {len(ground_truth_segments.labels())}")
    # Segment the audio
    # segments, _ = load_and_segment_audio(audio_path, segment_length=segment_length, overlap=overlap)
    segments, _ = segment_audio(audio, sr, segment_length=segment_length, overlap=overlap)
    # DEBUG LIMIT THE NUMBER OF SEGMENTS FOR TESTING
    # segments = segments[:50000]
    num_segments = len(segments)
    print(f"Number of segments: {num_segments}")
    # Apply VAD to remove non-speech segments
    segments = apply_vad(segments, sr, hp.vad_mode)
    print(f"Number of speech segments: {len(segments)}")
    # Extract d-vectors from the segments
    d_vectors = []
    for start_time, segment in segments:
        features = extract_features(segment, sr)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        d_vector = model(features)
        d_vectors.append((start_time, d_vector.cpu().detach().numpy()))

    print(f"Number of d-vectors: {len(d_vectors)}")

    # Aggregate d-vectors into 400ms segments
    aggregated_d_vectors = []
    current_segment_d_vectors = []
    current_segment_start_time = d_vectors[0][0]
    current_segment_duration = 0
    for start_time, d_vector in d_vectors:
        current_segment_d_vectors.append(d_vector)
        current_segment_duration += segment_length - overlap

        if current_segment_duration >= target_segment_length:
            # Average d-vectors for the segment
            aggregated_d_vectors.append((current_segment_start_time, average_d_vectors(
                np.concatenate(current_segment_d_vectors, axis=0))))  # Store start time with averaged d-vector
            # Reset for the next segment
            current_segment_d_vectors = []
            current_segment_start_time = start_time
            current_segment_duration = 0

    # Average any remaining d-vectors
    if current_segment_d_vectors:
        aggregated_d_vectors.append(
            (current_segment_start_time, average_d_vectors(np.concatenate(current_segment_d_vectors, axis=0))))

    # Extract d-vectors from aggregated_d_vectors for spectral clustering
    start_times, d_vectors = zip(*aggregated_d_vectors)  # Unpack start times and d-vectors
    d_vectors = np.array(d_vectors)  # Convert d-vectors to a numpy array

    print(f"Number of aggregated d-vectors: {len(d_vectors)}")

    # Clustering
    labels = spectral_clustering(d_vectors, sigma=sigma, percentile=percentile, random_state=seed, n_clusters=num_speakers)
    # labels = offline_kmeans(d_vectors, max_clusters=10, random_state=0)

    print(f"Number of speakers: {len(np.unique(labels))}")
    print(f"Speaker labels: {labels}")

    # Associate speaker labels with start times
    speaker_segments = []
    for i in range(len(labels)):
        start_time = start_times[i]
        end_time = start_time + target_segment_length / 1000
        speaker_id = labels[i]
        speaker_segments.append((start_time, end_time, speaker_id))

    print("Number of speakers after filtering: ", len(np.unique([s[2] for s in speaker_segments])))

    # groups consecutive segments with the same speaker
    speaker_segments = group_segments(speaker_segments)

    speaker_segments_annotations = Annotation()
    # Print the diarization results
    print("Speaker Diarization Results:")
    for start_time, end_time, speaker_id in speaker_segments:
        print(f"Speaker {speaker_id}: {start_time:.2f} - {end_time:.2f}")
        speaker_segments_annotations[Segment(start_time, end_time)] = f"spk{speaker_id:02d}"

    # evaluate the diarization
    der = eval_diarization(speaker_segments_annotations, ground_truth_segments)

    # save both the ground truth and predicted segments to rttm files
    annotation_to_rttm(speaker_segments_annotations, "predicted", output_path="predicted.rttm")
    annotation_to_rttm(ground_truth_segments, "ground_truth", output_path="ground_truth.rttm")

    # Visualize the ground truth and predicted speaker segments
    visualize_diarization(audio, sr, ground_truth_segments, speaker_segments_annotations)

    return speaker_segments, der


def group_segments(speaker_segments):
    # group consecutive segments with the same speaker, accommodating gaps with smoothing
    SMOOTHING_FACTOR = 0.2  # Maximum gap size to tolerate
    speaker_segments_grouped = []
    for start_time, end_time, speaker_id in speaker_segments:
        # Check if it's the same speaker AND if the segments are close enough (or overlapping)
        if (speaker_segments_grouped
                and speaker_segments_grouped[-1][2] == speaker_id  # Same speaker
                and start_time - speaker_segments_grouped[-1][1] <= SMOOTHING_FACTOR  # Close enough
        ):
            # Extend the existing group's end time
            speaker_segments_grouped[-1] = (
                speaker_segments_grouped[-1][0], max(end_time, speaker_segments_grouped[-1][1]), speaker_id)
        else:
            # Start a new group
            speaker_segments_grouped.append((start_time, end_time, speaker_id))
    speaker_segments = speaker_segments_grouped
    return speaker_segments


if __name__ == "__main__":
    hp = Hyperparameters()
    model_path = "500_vox_10_50_epoch.pth"
    if hp.eval_dataset == "data_conv":
        audio_path = "data_conv/eval/voxconverse_test_wav"
        random.seed(124)
        audio_files = []
        for audio_file in os.listdir(audio_path):
            if audio_file.endswith(".wav"):
                audio_files.append(os.path.join(audio_path, audio_file))

        der_list = []

        for i in range(1, 11):
            # randomly select a file
            file = random.choice(audio_files)
            print(f"Running diarization on {file}")
            _, der = run_diarization(file, model_path, sigma=3, percentile=95)
            der_list.append(der)

        # average the der
        average_der = np.mean(der_list)
        print(f"Average DER: {average_der:.2%}")
    elif hp.eval_dataset == "data_vox":
        der_list = []
        for i in range(0, 10):
            audio_path = "mixed_audio.wav"
            _, der = run_diarization(audio_path, model_path, seed=i, sigma=5, percentile=95)
            der_list.append(der)
        # average the der
        average_der = np.mean(der_list)
        print(f"Average DER: {average_der:.2%}")

    elif hp.eval_dataset == "data_call":
        audio_path = "data_call/index.duckdb"
        der_list = []
        for i in range(0, 10):
            random.seed(i)
            n = random.randint(0, 139)
            _, der = run_diarization(audio_path, model_path, callhome_index=n, sigma=0.7, percentile=95)
            der_list.append(der)
        # average the der
        average_der = np.mean(der_list)
        print(f"Average DER: {average_der:.2%}")
        print(f"DERs: {der_list}")
    else:
        print("Invalid dataset")
