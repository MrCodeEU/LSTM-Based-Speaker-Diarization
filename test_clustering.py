import os
import random
import librosa
import numpy as np
import soundfile as sf
import torch
#from datasets import load_dataset
from pyannote.core import Annotation, Segment
from clustering import spectral_clustering, offline_kmeans, dbscan_clustering, agglomerative_clustering
from config import Hyperparameters
from eval import eval_diarization, annotation_to_rttm
from extract_features import extract_features
from model import LSTMModel
from preprocessing import apply_vad, segment_audio
from utils import create_mixed_audio, load_audio_and_rttm, load_dataset_callhome
from visualize import visualize_diarization
import csv
import datetime


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
                    target_segment_length=400, seed=None, sigma=6.5, percentile=60, callhome_index=0, visualise=True):
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

    if hp.eval_dataset == "data_icsi" and audio_path != "mixed_audio.wav":
        # Load the audio and extract log mel features
        audio, sr = librosa.load(audio_path, sr=None)
    elif hp.eval_dataset == "data_vox" or audio_path == "mixed_audio.wav":
        # load n random speakers m audio files and splice them together to create a new audio file
        n = random.randint(2, 5)
        m = random.randint(2, 15)
        audio, sr, ground_truth_segments = create_mixed_audio("data_vox", n, m, seed=seed)

        # save the audio to a file
        audio_path_save = "mixed_audio.wav"
        sf.write(audio_path_save, audio, sr)
    elif hp.eval_dataset == "data_conv" and audio_path != "mixed_audio.wav":
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

    # Dictionary of clustering algorithms and their functions
    clustering_algorithms = {
        'Spectral': lambda: spectral_clustering(d_vectors, sigma=sigma, percentile=percentile, random_state=seed,
                                                visualizeCluster=True, visualize=False),
        'K-means': lambda: offline_kmeans(d_vectors, max_clusters=10, random_state=0),
        'DBSCAN': lambda: dbscan_clustering(d_vectors, eps=7, min_samples=20),
        'Agglomerative': lambda: agglomerative_clustering(d_vectors)
    }

    results = {}

    for algo_name, clustering_func in clustering_algorithms.items():
        print(f"\nProcessing {algo_name} clustering:")

        # Execute the clustering function
        labels = clustering_func()

        print(f"Number of speakers: {len(np.unique(labels))}")
        # print(f"Speaker labels: {labels}")

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
        print(f"{algo_name} Speaker Diarization Results:")
        for start_time, end_time, speaker_id in speaker_segments:
            # print(f"Speaker {speaker_id}: {start_time:.2f} - {end_time:.2f}")
            speaker_segments_annotations[Segment(start_time, end_time)] = f"spk{speaker_id:02d}"

        # evaluate the diarization
        metrics = eval_diarization(speaker_segments_annotations, ground_truth_segments)

        # save both the ground truth and predicted segments to rttm files
        annotation_to_rttm(speaker_segments_annotations, f"{algo_name}_predicted",
                           output_path=f"{algo_name}_predicted.rttm")
        annotation_to_rttm(ground_truth_segments, "ground_truth", output_path="ground_truth.rttm")

        # Visualize the ground truth and predicted speaker segments
        if visualise:
            visualize_diarization(audio, sr, ground_truth_segments, speaker_segments_annotations, algo_name)

        results[algo_name] = {
            'segments': speaker_segments,
            'metrics': metrics
        }

    return results, len(ground_truth_segments.labels())


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
    model_path = "500_vox_10_early_stop.pth"
    if hp.eval_dataset == "data_conv" or True:
        print("Running diarization on VoxConverse test set")
        audio_path = "data_conv/eval/voxconverse_test_wav"
        random.seed(22)
        audio_files = []
        for audio_file in os.listdir(audio_path):
            if audio_file.endswith(".wav"):
                audio_files.append(os.path.join(audio_path, audio_file))

        results_list = []

        for i in range(1, 31):
            file = random.choice(audio_files)
            print(f"Running diarization on {file}")
            results, actual_speakers = run_diarization(file, model_path, sigma=5, percentile=90)

            for algo_name, algo_results in results.items():
                predicted_speakers = len(set(segment[2] for segment in algo_results['segments']))
                speaker_diff = predicted_speakers - actual_speakers

                results_list.append({
                    'file': os.path.basename(file),
                    'algorithm': algo_name,
                    'der': algo_results['metrics'][0],
                    'jer': algo_results['metrics'][1],
                    'purity': algo_results['metrics'][2],
                    'coverage': algo_results['metrics'][3],
                    'completeness': algo_results['metrics'][4],
                    'ier': algo_results['metrics'][5],
                    'speaker_diff': speaker_diff
                })

            # Save the results in a CSV file with current timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        with open(f"diarization_results_{timestamp}.csv", mode='w', newline='') as file:
            fieldnames = ['file', 'algorithm', 'der', 'jer', 'purity', 'coverage', 'completeness', 'ier',
                          'speaker_diff']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_list:
                writer.writerow(result)

        # Calculate and print average metrics for each algorithm
        algorithms = set(result['algorithm'] for result in results_list)
        for algo in algorithms:
            algo_results = [r for r in results_list if r['algorithm'] == algo]
            avg_metrics = {
                'der': np.mean([r['der'] for r in algo_results]),
                'jer': np.mean([r['jer'] for r in algo_results]),
                'purity': np.mean([r['purity'] for r in algo_results]),
                'coverage': np.mean([r['coverage'] for r in algo_results]),
                'completeness': np.mean([r['completeness'] for r in algo_results]),
                'ier': np.mean([r['ier'] for r in algo_results]),
                'speaker_diff': np.mean([abs(r['speaker_diff']) for r in algo_results])
            }
            print(f"\nAverage metrics for {algo}:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.2%}")
    elif hp.eval_dataset == "data_vox":
        der_list = []
        for i in range(0, 10):
            audio_path = "mixed_audio.wav"
            _, der = run_diarization(audio_path, model_path, seed=i, sigma=8, percentile=80)
            der_list.append(der)
        # average the der
        average_der = np.mean(der_list)
        print(f"Average DER: {average_der:.2%}")

    elif hp.eval_dataset == "data_call":
        audio_path = "data_call/index.duckdb"
        der_list = []
        for i in range(0, 10):
            random.seed(i+2)
            n = random.randint(0, 139)
            _, der = run_diarization(audio_path, model_path, callhome_index=n, sigma=5, percentile=95, num_speakers=2)
            der_list.append(der)
        # average the der
        average_der = np.mean(der_list)
        print(f"Average DER: {average_der:.2%}")
        print(f"DERs: {der_list}")
    else:
        print("Invalid dataset")
