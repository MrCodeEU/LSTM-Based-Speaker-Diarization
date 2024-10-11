import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Hyperparameters
from run import run_diarization
from utils import create_mixed_audio


def main():
    hp = Hyperparameters()
    model_path = "mel_128_40.pth"
    if hp.train_eval_dataset == "data_conv":
        audio_path = "data_conv/eval/voxconverse_test_wav"
        random.seed(50)
        audio_files = [os.path.join(audio_path, "lubpm.wav"), os.path.join(audio_path, "eazeq.wav")]

        # Parameter Optimization
        best_der = float("inf")
        best_params = None
        der_values = {}  # Initialize dictionary to store DER values

        for sigma in np.arange(0, 6.0, 0.5):
            for percentile in np.arange(80, 99, 3):
                total_der = 0.0
                num_files_processed = 0

                for file in audio_files:
                    print(f"Running diarization on {file} with sigma={sigma}, percentile={percentile}")
                    _, der = run_diarization(file, model_path, seed=50, sigma=sigma, percentile=percentile, visualise=False)
                    total_der += der[0]
                    num_files_processed += 1

                    # if der > best_der:
                    #    break

                average_der = total_der / num_files_processed if num_files_processed > 0 else float("inf")

                if average_der < best_der:
                    best_der = average_der
                    best_params = (sigma, percentile)

                der_values[(sigma, percentile)] = average_der  # Store DER value

        print(f"Best DER: {best_der} achieved with sigma={best_params[0]}, percentile={best_params[1]}")

        # Create heatmap
        sigmas = np.unique([k[0] for k in der_values.keys()])
        percentiles = np.unique([k[1] for k in der_values.keys()])
        der_matrix = np.array([[der_values.get((s, p), np.nan) for p in percentiles] for s in sigmas])

        plt.figure(figsize=(10, 8))
        plt.imshow(der_matrix, cmap="viridis", origin="lower", interpolation="nearest")
        plt.colorbar(label="DER")
        plt.yticks(range(len(sigmas)), sigmas)
        plt.xticks(range(len(percentiles)), percentiles)
        plt.ylabel("Sigma")
        plt.xlabel("Percentile")
        plt.title("DER Heatmap for Sigma and Percentile")
        plt.show()

    elif hp.train_eval_dataset == "data_call":
        audio_path = "data_call/index.duckdb"
        random.seed(50)

        # Parameter Optimization
        best_der = float("inf")
        best_params = None
        der_values = {}  # Initialize dictionary to store DER values

        for sigma in tqdm(np.arange(1, 15.0, 2), desc="Optimizing Sigma", unit="sigma"):
            for percentile in np.arange(70, 99, 3):
                total_der = 0.0
                num_files_processed = 0

                for i in range(0, 2):
                    n = random.randint(0, 139)
                    _, der = run_diarization(audio_path, model_path, callhome_index=n, num_speakers=2, visualise=False, sigma=sigma, percentile=percentile)
                    total_der += der[0]
                    num_files_processed += 1

                    # if der > best_der:
                    #    break

                average_der = total_der / num_files_processed if num_files_processed > 0 else float("inf")

                if average_der < best_der:
                    best_der = average_der
                    best_params = (sigma, percentile)

                der_values[(sigma, percentile)] = average_der  # Store DER value

        print(f"Best DER: {best_der} achieved with sigma={best_params[0]}, percentile={best_params[1]}")

        # Create heatmap
        sigmas = np.unique([k[0] for k in der_values.keys()])
        percentiles = np.unique([k[1] for k in der_values.keys()])
        der_matrix = np.array([[der_values.get((s, p), np.nan) for p in percentiles] for s in sigmas])

        plt.figure(figsize=(10, 8))
        plt.imshow(der_matrix, cmap="viridis", origin="lower", interpolation="nearest")
        plt.colorbar(label="DER")
        plt.yticks(range(len(sigmas)), sigmas)
        plt.xticks(range(len(percentiles)), percentiles)
        plt.ylabel("Sigma")
        plt.xlabel("Percentile")
        plt.title("DER Heatmap for Sigma and Percentile")
        plt.show()
    elif hp.train_eval_dataset == "data_vox" or True:
        audio_path = "mixed_audio.wav"
        random.seed(50)

        # Parameter Optimization
        best_der = float("inf")
        best_params = None
        der_values = {}  # Initialize dictionary to store DER values

        for sigma in tqdm(np.arange(1, 15.0, 2), desc="Optimizing Sigma", unit="sigma"):
            for percentile in np.arange(70, 99, 3):
                total_der = 0.0
                num_files_processed = 0

                audio_files = []
                for i in range(1, 3):
                    print(f"Running diarization on {i} with sigma={sigma}, percentile={percentile}")
                    _, der = run_diarization(audio_path, model_path, seed=50, sigma=sigma, percentile=percentile, visualise=False)
                    total_der += der[0]
                    num_files_processed += 1

                    # if der > best_der:
                    #    break

                average_der = total_der / num_files_processed if num_files_processed > 0 else float("inf")

                if average_der < best_der:
                    best_der = average_der
                    best_params = (sigma, percentile)

                der_values[(sigma, percentile)] = average_der  # Store DER value

        print(f"Best DER: {best_der} achieved with sigma={best_params[0]}, percentile={best_params[1]}")

        # Create heatmap
        sigmas = np.unique([k[0] for k in der_values.keys()])
        percentiles = np.unique([k[1] for k in der_values.keys()])
        der_matrix = np.array([[der_values.get((s, p), np.nan) for p in percentiles] for s in sigmas])

        plt.figure(figsize=(10, 8))
        plt.imshow(der_matrix, cmap="viridis", origin="lower", interpolation="nearest")
        plt.colorbar(label="DER")
        plt.yticks(range(len(sigmas)), sigmas)
        plt.xticks(range(len(percentiles)), percentiles)
        plt.ylabel("Sigma")
        plt.xlabel("Percentile")
        plt.title("DER Heatmap for Sigma and Percentile")
        plt.savefig("heatmap.png")
        plt.show()


if __name__ == "__main__":
    main()
