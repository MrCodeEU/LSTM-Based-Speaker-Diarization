# # Specify the number of segments to display
#         num_segments = 4000
#
#         # Get the end time of the last displayed audio segment
#         end_time = segments[num_segments - 1][0] + len(segments[num_segments - 1][1]) / sr
#
#         # Visualize audio segments and speech segments
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
#
#         # Plot audio segments and waveforms
#         for segment in segments[:num_segments]:
#             start_time, audio_segment = segment
#             segment_end_time = start_time + len(audio_segment) / sr
#             ax1.plot([start_time, segment_end_time], [0, 0], color='b', linewidth=2)
#
#             # Plot audio waveform for each segment
#             time = np.linspace(start_time, segment_end_time, len(audio_segment))
#             ax1.plot(time, audio_segment, color='gray', linewidth=0.5)
#
#         ax1.set_title(f"First {num_segments} Audio Segments")
#         ax1.set_ylabel("Amplitude")
#         ax1.set_ylim(-1, 1)
#
#         # Plot speech segments within the time range of displayed audio segments
#         for segment in speech_segments:
#             start_time, speech_segment = segment
#             segment_end_time = start_time + len(speech_segment) / sr
#             if start_time <= end_time:
#                 ax2.plot([start_time, segment_end_time], [0, 0], color='g', linewidth=2)
#
#                 # Plot audio waveform for each speech segment
#                 time = np.linspace(start_time, segment_end_time, len(speech_segment))
#                 ax2.plot(time, speech_segment, color='gray', linewidth=0.5)
#
#         ax2.set_title("Speech Segments")
#         ax2.set_ylabel("Amplitude")
#         ax2.set_ylim(-1, 1)
#
#         # Set common labels and display the plot
#         plt.xlabel("Time (seconds)")
#         plt.xlim(0, end_time)
#         plt.tight_layout()
#
# # Visualize audio and segments
# librosa.display.waveshow(audio, sr=sr, ax=ax3)
# ax3.set_title(f"Audio of {meeting_dir}")
# ax3.set_xlabel("Time")
# ax3.set_ylabel("Amplitude")
#
# # Plot the segments where every participant has a different color
# for i, (start_time, end_time, participant) in enumerate(segments_from_xml):
#     ax3.add_patch(patches.Rectangle((start_time, -1), end_time - start_time, 2,
#                                     color=plt.cm.tab20(i % 20), alpha=0.5))
#
# # add a legend for the unique participant ids by filtering the segments for unique participant ids
# unique_participants = set([participant for _, _, participant in segments_from_xml])
# ax3.legend(handles=[patches.Patch(color=plt.cm.tab20(i % 20), label=participant)
#                     for i, participant in enumerate(unique_participants)])
# plt.show()
import librosa
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pyannote.core import Annotation


def visualize_affinity_matrix_refinement(A, step_name, sigma=None, percentile=None):
    """Visualizes the affinity matrix at a given refinement step."""

    plt.figure(figsize=(8, 8))

    # Add step-specific information to the title
    if step_name == "Gaussian Blur" and sigma is not None:
        plt.title(f"{step_name} (sigma={sigma})")
    elif step_name == "Row-wise Thresholding" and percentile is not None:
        plt.title(f"{step_name} ({percentile}th Percentile)")
    else:
        plt.title(step_name)

    plt.imshow(A, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()


def plot_spectrograms(log_mel_spectrograms, labels):
    """Plots log mel spectrograms for each unique speaker as subplots."""

    unique_labels = set(labels)
    num_speakers = len(unique_labels)

    # Calculate rows and columns for subplots
    rows = int(np.ceil(np.sqrt(num_speakers)))
    cols = int(np.ceil(num_speakers / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))

    for i, label in enumerate(unique_labels):
        # Find the index of the first occurrence of the label
        indices = np.where(labels == label)[0]  # this line is the fixed version
        log_mel_spectrogram = None
        if len(indices) > 0:
            index = indices[0]
            log_mel_spectrogram = log_mel_spectrograms[index]

        # Get the correct subplot axis
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        ax.imshow(log_mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"Speaker {label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
        fig.colorbar(ax.images[0], ax=ax, label="dB")

    # Hide any unused subplots
    for i in range(num_speakers, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()


def get_speaker_color(speaker, color_map, available_colors):
    if speaker not in color_map:
        color_map[speaker] = available_colors.pop(0)
    return color_map[speaker]


def get_speaker_color(speaker, color_map, available_colors):
    if speaker not in color_map:
        color_map[speaker] = available_colors.pop(0)
    return color_map[speaker]


def visualize_diarization(y, sr, ground_truth_segments: Annotation, speaker_segments_annotations: Annotation):
    # Load audio file
    duration = librosa.get_duration(y=y, sr=sr)

    # Create the plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot the audio waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    ax[0].set(title='Audio Waveform')

    # Prepare color maps
    ground_truth_color_map = {}
    predicted_color_map = {}
    ground_truth_colors = [cm.get_cmap("tab20")(i) for i in range(20)]
    predicted_colors = [cm.get_cmap("tab20")(i) for i in range(20)]

    # Plot ground truth speaker segments
    for segment in ground_truth_segments.itersegments():
        start, end = segment
        speaker = ground_truth_segments[segment]
        color = get_speaker_color(speaker, ground_truth_color_map, ground_truth_colors)
        ax[1].axvspan(start, end, color=color, alpha=0.5, lw=0)

    ax[1].set(title='Ground Truth Speaker Segments')

    # Plot predicted speaker segments
    for segment in speaker_segments_annotations.itersegments():
        start, end = segment
        speaker = speaker_segments_annotations[segment]
        color = get_speaker_color(speaker, predicted_color_map, predicted_colors)
        ax[2].axvspan(start, end, color=color, alpha=0.5, lw=0)

    ax[2].set(title='Predicted Speaker Segments')

    # Set x-axis labels
    ax[2].set_xlabel('Time (s)')

    # Set y-axis labels
    for a in ax:
        a.set_ylabel('Amplitude')

    # Add legends
    ground_truth_handles = [mpatches.Patch(color=ground_truth_color_map[speaker], label=speaker) for speaker in
                            ground_truth_color_map]
    predicted_handles = [mpatches.Patch(color=predicted_color_map[speaker], label=speaker) for speaker in
                         predicted_color_map]

    ax[1].legend(handles=ground_truth_handles, loc='upper right')
    ax[2].legend(handles=predicted_handles, loc='upper right')

    plt.tight_layout()
    plt.show()