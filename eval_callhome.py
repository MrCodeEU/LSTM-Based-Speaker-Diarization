import duckdb
import numpy as np
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment

import run
import visualize
from config import Hyperparameters

# Read DuckDB file
con = duckdb.connect('data_call/index.duckdb')
query = "SELECT * FROM data"
df = con.execute(query).fetchdf()

audi_with_speakers = {}

for i in range(len(df)):
    # Get audio bytes
    audio_bytes = df['audio'][i].get("bytes")
    start_times = df['timestamps_start'][i]
    end_times = df['timestamps_end'][i]
    speakers = df['speakers'][i]

    # Determine audio parameters (adjust these based on your actual audio format)
    sample_width = 2  # 2 bytes per sample (16-bit PCM)
    num_channels = 1  # Mono audio
    sample_rate = 16000  # Common sample rate

    # Convert bytes to NumPy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16 / (2**(8*sample_width - 1))  # Normalize to [-1, 1]

    # Create an Annotation object
    annotation = Annotation()
    for start, end, speaker in zip(start_times, end_times, speakers):
        annotation[Segment(start, end)] = speaker

    audi_with_speakers[i] = (audio_float, annotation)


hp = Hyperparameters()
hp.eval_dataset = "data_call"

run.run_diarization()


con.close()
