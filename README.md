# LSTM-based Speaker Diarization System

This repository contains the implementation of an advanced LSTM-based speaker diarization system as described in the paper "Optimizing LSTM-based Speaker Diarization: Comparing Architectures and Clustering Methods" by Michael Reinegger.

![Diarization Example](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/diarization_example.png)

## Overview

Speaker diarization is the process of partitioning an audio stream into homogeneous segments according to the speaker's identity. This system uses Long Short-Term Memory (LSTM) neural networks to generate d-vectors from audio features, which are then clustered to identify unique speakers.

Key features of this implementation include:
- LSTM and BiLSTM architectures for d-vector generation
- Multiple clustering algorithms for speaker segmentation
- Comprehensive hyperparameter tuning
- Extensive experimental analysis

## System Architecture

![System Architecture](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/system_architecture.png)

1. **Feature Extraction**: Mel-spectrograms are extracted from 30ms audio segments.
2. **Voice Activity Detection (VAD)**: WebRTC VAD is used to filter out non-speech segments.
3. **Neural Network**: LSTM/BiLSTM network generates d-vectors from speech features.
4. **Clustering**: Various algorithms (Spectral, DBSCAN, K-Means, Agglomerative) group d-vectors into speaker clusters.
5. **Post-processing**: Short segments are merged and final diarization output is generated.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- librosa 0.8+
- WebRTC VAD
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- pyannotate

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare your audio datasets in the following structure:

```
data/
├── voxceleb1/
├── voxconverse/
└── callhome/
```

### Training

To train the model:

```bash
python train.py
```

You can modify training parameters in `config.py`.

### Inference

To run inference on new audio:

```bash
python run.py path/to/model.pth path/to/audio.wav
```

## Configuration

Adjust hyperparameters in `config.py`:

```python
# Neural Network Architecture
input_size = 40
hidden_size = 768
num_layers = 3
projection_size = 256

# Training Parameters
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# Data Preprocessing
min_num_speakers = 2
max_num_speakers = 8
num_segments = 10
```

## Experiments and Results

### 1. Hyperparameter Optimization

We conducted extensive experiments to optimize key hyperparameters:

#### Spectral Clustering Parameters

![Spectral Clustering Heatmap](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/spectral_clustering_heatmap.png)

Optimal parameters: σ = 7, percentile = 85%

#### Mel Filter Bank Settings

![Mel Filter Bank Heatmap](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/mel_filter_bank_heatmap.png)

Best configuration: hop-length = 40ms, 128 FFTs

### 2. Clustering Method Comparison

We compared four clustering algorithms:

![Clustering Comparison](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/clustering_comparison.png)

Spectral Clustering and Agglomerative Clustering showed the best overall performance.

### 3. Neural Network Architecture Analysis

Various LSTM and BiLSTM configurations were tested:

![Neural Net Comparison](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/neural_net_comparison.png)

The 3-layer LSTM with 768 nodes per layer achieved the best balance of performance and computational efficiency.

## Performance

The system achieves an average Diarization Error Rate (DER) of 20% on the Vox Converse dataset.

![DER Results](https://raw.githubusercontent.com/yourusername/your-repo-name/main/images/der_results.png)

## Contributing

We welcome contributions to improve the system. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@techreport{reinegger2024optimizing,
  title={Optimizing LSTM-based Speaker Diarization: Comparing Architectures and Clustering Methods},
  author={Reinegger, Michael},
  year={2024},
  institution={Johannes Kepler University Linz},
  note={Available at: https://www.michaelreinegger.com/publications/speaker-diarization-lstm.pdf}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VoxCeleb and VoxConverse datasets
- WebRTC Project for the VAD implementation
- PyTorch team for the deep learning framework