import glob
import os
import random
from multiprocessing import Pool

import duckdb
import librosa
import numpy as np
from pyannote.core import Annotation, Segment
from tqdm import tqdm
import visualize
from preprocessing import process_segment, segment_audio, load_segments


def binary_search(segments_from_xml, start_time):
    low, high = 0, len(segments_from_xml) - 1
    while low <= high:
        mid = (low + high) // 2
        start_time_xml, end_time_xml, participant = segments_from_xml[mid]
        if start_time_xml <= start_time <= end_time_xml:
            return participant
        elif start_time < start_time_xml:
            high = mid - 1
        else:
            low = mid + 1
    return None


def assign_segments_to_speakers(audio_segments, xml_segments, sr):
    """Assigns audio segments to speakers based on XML segment information."""
    speech_segments = {}  # Speaker-wise segments
    for start_time, audio_segment in audio_segments:
        for start_time_xml, end_time_xml, participant in xml_segments:
            if start_time_xml <= start_time <= end_time_xml:
                speech_segments.setdefault(participant, []).append((start_time, audio_segment))
                break  # Found matching XML segment
        else:
            speech_segments.setdefault("unknown", []).append((start_time, audio_segment))
    if "unknown" in speech_segments:
        del speech_segments["unknown"]  # Optionally remove "unknown" speaker
    return speech_segments, sr


def process_segments_parallel(speech_segments, sr):
    # feature extraction
    log_mel_spectrograms = []
    labels = []
    # Prepare arguments for multiprocessing
    args = []
    for participant, segments in speech_segments.items():
        for start_time, audio_segment in segments:
            args.append((participant, (start_time, audio_segment), sr))
    # Create a multiprocessing Pool
    with Pool() as p:
        results = list(
            tqdm(p.imap(process_segment, args), total=len(args), desc="Processing segments", unit="segment"))
    # Separate the results into log_mel_spectrograms and labels
    for log_mel_spectrogram, participant in results:
        log_mel_spectrograms.append(log_mel_spectrogram)
        labels.append(participant)
    return labels, log_mel_spectrograms


def load_rttm_segments(rttm_file):
    segments = []
    with open(rttm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 10:
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speaker_id = parts[7]
                segments.append((start_time, end_time, speaker_id))
    return segments


def assign_segments_to_speakers_rttm(audio_segments, rttm_segments, sr):
    speech_segments = {}
    overlap_segments = set()  # To keep track of overlapping segments

    for start_time, audio_segment in tqdm(audio_segments, desc="Assigning segments to speakers", unit="segment"):
        participant = binary_search_rttm(rttm_segments, start_time)
        if participant:
            if start_time in overlap_segments:
                # This segment has already been assigned to another participant
                continue
            if participant in speech_segments and any(start_time == s for s, _ in speech_segments[participant]):
                # This segment is already assigned to this participant
                overlap_segments.add(start_time)
                # Remove the segment from all participants it was assigned to
                for p in speech_segments:
                    speech_segments[p] = [(s, seg) for s, seg in speech_segments[p] if s != start_time]
            else:
                speech_segments.setdefault(participant, []).append((start_time, audio_segment))
        else:
            speech_segments.setdefault("unknown", []).append((start_time, audio_segment))

    if "unknown" in speech_segments:
        del speech_segments["unknown"]

    return speech_segments, sr


# Helper function to perform binary search (example implementation)
def binary_search_rttm(rttm_segments, start_time):
    # Assuming rttm_segments is sorted by start_time
    low, high = 0, len(rttm_segments) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_start, mid_end, participant = rttm_segments[mid]
        if mid_start <= start_time < mid_end:
            return participant
        elif start_time < mid_start:
            high = mid - 1
        else:
            low = mid + 1
    return None


def extract_and_label_features_voxconverse(hp, train_dir, segments_dir, sr=16000):
    if not os.path.exists(hp.dataset + "/log_mel_spectrograms.npy") and not os.path.exists(hp.dataset + "/labels.npy"):
        train_dir = os.path.join(train_dir, "audio")
        speech_segments = {}
        for audio_file in os.listdir(train_dir):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(train_dir, audio_file)
                rttm_file = os.path.join(os.path.join(segments_dir, "dev"), audio_file.replace(".wav", ".rttm"))

                audio = librosa.load(audio_path, sr=sr)[0]
                target_rms = 0.1
                audio = librosa.util.normalize(audio, norm=np.inf, threshold=target_rms)
                segments, sr = segment_audio(audio, sr)

                rttm_segments = load_rttm_segments(rttm_file)
                rttm_segments.sort(key=lambda x: x[0])  # Sort by the start time

                speech_segments_id, sr = assign_segments_to_speakers_rttm(segments, rttm_segments, sr)

                for participant, segments in speech_segments_id.items():
                    speech_segments.setdefault(participant, []).extend(segments)

        # DEBUG LIMIT THE NUMBER OF SEGMENTS FOR TESTING
        # speech_segments = {participant: segments[:5000] for participant, segments in speech_segments.items()}

        print("Number of participants: ", len(speech_segments))
        for participant, segments in speech_segments.items():
            print(f"Number of segments for {participant}: {len(segments)}")

        labels, log_mel_spectrograms = process_segments_parallel(speech_segments, sr)

        log_mel_spectrograms = np.array(log_mel_spectrograms)

        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

        np.save(hp.dataset + "/log_mel_spectrograms.npy", log_mel_spectrograms)
        np.save(hp.dataset + "/labels.npy", labels)
    else:
        log_mel_spectrograms = np.load(hp.dataset + "/log_mel_spectrograms.npy")
        labels = np.load(hp.dataset + "/labels.npy")
        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

    visualize.plot_spectrograms(log_mel_spectrograms, labels)

    return labels, log_mel_spectrograms


def extract_and_label_features_voxceleb(hp, train_dir, sr=16000):
    log_mel_spectrograms = None
    if not os.path.exists(hp.dataset + "/log_mel_spectrograms.npy") and not os.path.exists(hp.dataset + "/labels.npy"):
        # Iterate over speaker directories and get the audio files per speaker recursively
        audio_files = {}
        for speaker_id in os.listdir(train_dir):
            speaker_path = os.path.join(train_dir, speaker_id)

            # recursively get all audio files for the speaker
            audio_files_id = glob.glob(speaker_path + "/**/*.wav", recursive=True)
            speaker_id = int(speaker_id.replace("id1", ""))
            audio_files[speaker_id] = audio_files_id

        # DEBUG LIMIT THE NUMBER OF audio_files FOR TESTING per speaker to n random audio files
        # n = 20
        # audio_files = {speaker_id: random.sample(audio_files_id, n) for speaker_id, audio_files_id in audio_files.items()}

        # DEBUG LIMIT THE NUMBER OF SPEAKERS FOR TESTING to n random speakers
        n = 50
        audio_files = dict(random.sample(list(audio_files.items()), n))
        for speaker_id, audio_files_id in audio_files.items():
            print(f"Speaker: {speaker_id}, Number of audio files: {len(audio_files_id)}")

        # Load and segment the audio
        speech_segments = {}
        for speaker_id, audio_files_id in tqdm(audio_files.items(), desc="Segmenting audio", unit="speaker"):
            segments = []
            for audio_file in audio_files_id:
                # load audio
                audio, _ = librosa.load(audio_file, sr=sr)
                target_rms = 0.1
                audio = librosa.util.normalize(audio, norm=np.inf, threshold=target_rms)
                segments_id, sr = segment_audio(audio, sr)
                segments.extend(segments_id)
            speech_segments[speaker_id] = segments

        # extract features in parallel
        labels, log_mel_spectrograms = process_segments_parallel(speech_segments, sr)

        log_mel_spectrograms = np.array(log_mel_spectrograms)

        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

        # save the log mel spectrograms and labels to a file
        np.save(hp.dataset + "/log_mel_spectrograms.npy", log_mel_spectrograms)
        np.save(hp.dataset + "/labels.npy", labels)

    else:
        log_mel_spectrograms = np.load(hp.dataset + "/log_mel_spectrograms.npy")
        labels = np.load(hp.dataset + "/labels.npy")
        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

    # Visualize log mel spectrograms for each speaker
    visualize.plot_spectrograms(log_mel_spectrograms, labels)

    return labels, log_mel_spectrograms


def extract_and_label_features(hp, segments_dir, train_dir):
    # TODO name npy files after used hyperparameters
    if hp.dataset == "data_icsi":
        labels, log_mel_spectrograms = extract_and_label_features_icsi(hp, segments_dir, train_dir)
    elif hp.dataset == "data_vox":
        labels, log_mel_spectrograms = extract_and_label_features_voxceleb(hp, train_dir)
    else:
        labels, log_mel_spectrograms = extract_and_label_features_voxconverse(hp, train_dir, segments_dir)

    return labels, log_mel_spectrograms


def extract_and_label_features_icsi(hp, segments_dir, train_dir):
    if not os.path.exists(hp.dataset + "/log_mel_spectrograms.npy") and not os.path.exists(hp.dataset + "/labels.npy"):
        # Iterate over the meeting directories
        for meeting_dir in os.listdir(train_dir):
            audio_path = os.path.join(train_dir, meeting_dir, f"{meeting_dir}.interaction.wav")
            print(f"Processing audio file: {audio_path}")

            # load audio for later visualization
            audio = librosa.load(audio_path, sr=None)[0]

            # Load and segment the audio
            audio, _ = librosa.load(audio_path)
            segments, sr = segment_audio(audio, sr)

            # DEBUG LIMIT THE NUMBER OF SEGMENTS FOR TESTING
            # segments = segments[:50000]

            print(f"Number of segments: {len(segments)}")
            print(f"Segment length: {len(segments[0][1]) / sr:.2f} seconds")
            print(segments[0][1].shape)

            # Get all XML files that match the pattern
            xml_files = glob.glob(os.path.join(segments_dir, f"{meeting_dir}.*.segs.xml"))

            segments_from_xml = []

            # Iterate over the XML files
            for xml_file in xml_files:
                if os.path.isfile(xml_file):
                    # Load segments from the XML file
                    xml_segments = load_segments(xml_file)
                    print(f"Number of segments from {xml_file}: {len(xml_segments)}")

                    # Merge the segments
                    segments_from_xml.extend(xml_segments)
                else:
                    print(f"XML file not found: {xml_file}")

            print(f"Total number of segments: {len(segments_from_xml)}")

            # Sort segments_from_xml by start time
            segments_from_xml.sort(key=lambda x: x[0])  # Sort by the first element (start time)

            # Process segments using binary search
            speech_segments = {}
            for start_time, audio_segment in segments:
                participant = binary_search(segments_from_xml, start_time)
                if participant:
                    speech_segments.setdefault(participant, []).append((start_time, audio_segment))
                else:
                    speech_segments.setdefault("unknown", []).append((start_time, audio_segment))

            # remove unknown if not needed
            if "unknown" in speech_segments:
                del speech_segments["unknown"]
            print("Number of participants: ", len(speech_segments))
            # print the number of segments for each participant
            for participant, segments in speech_segments.items():
                print(f"Number of segments for {participant}: {len(segments)}")

        labels, log_mel_spectrograms = process_segments_parallel(speech_segments, sr)

        log_mel_spectrograms = np.array(log_mel_spectrograms)

        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

        # save the log mel spectrograms and labels to a file
        np.save(hp.dataset + "/log_mel_spectrograms.npy", log_mel_spectrograms)
        np.save(hp.dataset + "/labels.npy", labels)

    else:
        log_mel_spectrograms = np.load(hp.dataset + "/log_mel_spectrograms.npy")
        labels = np.load(hp.dataset + "/labels.npy")
        print(f"Number of log mel spectrograms: {len(log_mel_spectrograms)}")
        print(f"Shape of log mel spectrogram: {log_mel_spectrograms[0].shape}")

    # Visualize log mel spectrograms for each speaker
    visualize.plot_spectrograms(log_mel_spectrograms, labels)

    return labels, log_mel_spectrograms


def create_mixed_audio(data_dir, n_speakers, max_files_per_speaker, silence_duration=2, seed=150):
    """Creates a mixed audio file and corresponding ground truth annotations."""

    data_dir = os.path.join(data_dir, "eval")
    speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    random.seed(seed)
    selected_speakers = random.sample(speaker_dirs, n_speakers)

    mixed_audio = []
    ground_truth_segments = Annotation()
    current_time = 0.0  # Track the current time in the mixed audio

    for speaker in selected_speakers:
        speaker_path = os.path.join(data_dir, speaker)
        audio_files = glob.glob(speaker_path + "/**/*.wav", recursive=True)
        selected_files = random.sample(audio_files, min(len(audio_files), max_files_per_speaker))

        for file in selected_files:
            audio, sr = librosa.load(file, sr=None)
            segment_duration = len(audio) / sr
            ground_truth_segments[Segment(current_time, current_time + segment_duration)] = speaker
            current_time += segment_duration

            mixed_audio.append(audio)

            # Add silence segment and update current_time
            silence = np.zeros(int(silence_duration * sr))
            mixed_audio.append(silence)
            current_time += silence_duration

    # Concatenate and normalize
    mixed_audio = np.concatenate(mixed_audio)
    mixed_audio /= np.max(np.abs(mixed_audio))

    return mixed_audio, sr, ground_truth_segments


def load_audio_and_rttm(audio_path):
    """Loads audio and RTTM file for a given meeting."""
    audio, sr = librosa.load(audio_path, sr=None)
    # go two directories up for rttm file
    rttm_path = os.path.join(os.path.dirname(audio_path), "..", "..", "segments", "test",
                             f"{os.path.basename(audio_path).replace('.wav', '.rttm')}")
    ground_truth_segments_loaded = load_rttm_segments(rttm_path)
    # turn segments into annotation object
    ground_truth_segments = Annotation()
    for start_time, end_time, speaker_id in ground_truth_segments_loaded:
        ground_truth_segments[Segment(start_time, end_time)] = speaker_id
    return audio, sr, ground_truth_segments


def load_dataset_callhome(audio_path, index):
    # Read DuckDB file
    con = duckdb.connect(audio_path)
    query = "SELECT * FROM data"
    df = con.execute(query).fetchdf()

    audi_with_speakers = {}

    # Get audio bytes
    audio_bytes = df['audio'][index].get("bytes")
    start_times = df['timestamps_start'][index]
    end_times = df['timestamps_end'][index]
    speakers = df['speakers'][index]

    # Determine audio parameters (adjust these based on your actual audio format)
    sample_width = 2  # 2 bytes per sample (16-bit PCM)
    num_channels = 1  # Mono audio
    sample_rate = 16000  # Common sample rate

    # Convert bytes to NumPy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16 / (2 ** (8 * sample_width - 1))  # Normalize to [-1, 1]

    # cut off the first 1 second of the audio (since there is a large peak)
    audio_float = audio_float[int(sample_rate):]

    target_rms = 0.1
    audio_float = librosa.util.normalize(audio_float, norm=np.inf, threshold=target_rms)

    # Create an Annotation object
    annotation = Annotation()
    for start, end, speaker in zip(start_times, end_times, speakers):
        if speaker == "A":
            annotation[Segment(start, end)] = "spk00"
        elif speaker == "B":
            annotation[Segment(start, end)] = "spk01"

    con.close()
    return audio_float, sample_rate, annotation
