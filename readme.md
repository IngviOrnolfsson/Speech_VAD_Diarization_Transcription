````markdown
# Conversation VAD Labeler Pipeline

Automated end-to-end processing of dyadic conversation recordings. Starting from raw per-speaker audio files, the pipeline produces cleaned, merged, transcribed, and context-aware labeled segments ready for manual inspection or import into ELAN.

Key stages handled by the code in this repository:

1. **Voice Activity Detection (VAD)** using rVAD wrappers.
2. **Segment filtering** to drop short or low-energy detections.
3. **Windowed turn merging** across speakers with configurable heuristics.
4. **Whisper (Transformers) transcription** of merged segments with caching and batching safeguards.
5. **Entropy-based labeling** and context-aware merging to generate final annotations.

The main orchestration lives in the package module `conversation_vad_labeler.conversation` via
`process_conversation`, which is also exposed at the package root for convenience.

---

## Installation

```bash
git clone <your-repo-url>
cd conversation_vad_labeler_package

# Optional: create a dedicated environment
conda env create -f environment.yml
# Or if you prefer mamba
mamba env create -f environment.yml

pip install -e .
```

PyTorch wheels differ across platforms; if you prefer GPU acceleration follow the
[official PyTorch install selector](https://pytorch.org/get-started/locally/) before running
`pip install -e .`. The code automatically falls back to CPU inference when CUDA is not available.

---

## Repository tour

```
conversation_vad_labeler_package/
├── conversation_pipeline.py         # Thin CLI wrapper (imports package API)
├── conversation_vad_labeler/
│   ├── config.py                    # YAML configuration helpers
│   ├── vad.py                       # RVAD wrapper
│   ├── postprocess_vad.py           # Segment filtering utilities
│   ├── merge_turns.py               # Turn-merging logic
│   ├── transcription.py             # Whisper transcription helpers
│   ├── labeling.py                  # Entropy + context-based labeling
│   └── ...
├── examples/
│   └── recordings/                  # Sample audio/text resources
├── docs/                            # Protocol and documentation
├── requirements.txt / environment.yml
└── readme.md
```

---

## Quick start (Python API)

```python
from conversation_vad_labeler import process_conversation

speakers_audio = {
  "P1": "examples/recordings/EXP9_None_p1_trial2.wav",
  "P2": "examples/recordings/EXP9_None_p2_trial2.wav",
}

results = process_conversation(
  speakers_audio=speakers_audio,
  output_dir="outputs/example_run",
  vad_min_duration=0.07,
  energy_margin_db=10.0,
  gap_thresh=0.2,
  short_utt_thresh=0.7,
  window_sec=2.0,
  whisper_model_name="large-v3",
  whisper_device="auto",  # chooses CUDA when available otherwise CPU
  whisper_language="da",
  whisper_transformers_batch_size=100,
  batch_size=240.0,
)

print(results)
```

The returned dictionary contains both file paths and in-memory pandas DataFrames:

```python
{
  "output_dir": "outputs/example_run",
  "vad_paths": {"P1": "outputs/example_run/P1/..._vad.txt", ...},
  "merged_turns": "outputs/example_run/merged_turns.txt",
  "classified": "outputs/example_run/classified_transcriptions.txt",
  "final_labels": "outputs/example_run/final_labels.txt",
  "turns_df": <pandas.DataFrame>,
  "classified_df": <pandas.DataFrame>,
  "final_df": <pandas.DataFrame>,
}
```

These TSV outputs include timestamped speaker information and can be imported into ELAN or other annotation tools.

---

## Pipeline details

### 1. Voice Activity Detection
- `conversation_vad_labeler.vad.RVADWrapper.run_vad` processes each speaker file.
- Outputs tab-separated files per speaker (`*_vad.txt`).

### 2. Energy-based filtering
- Low-energy segments are dropped via `filter_low_energy_segments`.
- Results are concatenated and sorted globally; summary counts are printed.

### 3. Turn merging
- `create_turns_df_windowed` merges overlapping or closely spaced segments.
- Tunable parameters: `gap_thresh`, `short_utt_thresh`, `window_sec`, `merge_max_dur`, etc.
- Produces `merged_turns.txt` and a DataFrame used downstream.

### 4. Whisper transcription
- `load_whisper_model` loads `openai/whisper-<model_name>` through the Hugging Face Transformers pipeline.
- Segments per speaker are exported as WAV (16-bit PCM) plus text caches.
- Batches are built by total duration (`batch_size` seconds cap) while respecting the pipeline's `whisper_transformers_batch_size` clips per call.
- Memory-aware fallback retries segments individually when Hugging Face collation or CUDA OOM occurs.

### 5. Classification & context merging
- `classify_transcriptions` applies an entropy threshold to flag low-confidence/backchannel segments.
- `merge_turns_with_context` merges across adjacent turns using `max_backchannel_dur` and `max_gap_sec` limits.
- Outputs `classified_transcriptions.txt` and `final_labels.txt`.

---

## CLI example

If you prefer the script entry point, run:

```bash
python conversation_pipeline.py
```

The CLI wrapper calls `process_conversation` with the bundled recordings under
`examples/recordings/`. Override the paths or build your own CLI by importing the
package API directly.

---

## Tips

- Adjust `batch_size` (seconds) and `whisper_transformers_batch_size` (clips) to fit your GPU memory budget. Smaller values reduce the likelihood of `ValueError: ... different keys` from the Transformers data loader.
- Use `whisper_device="auto"` (default) to let the library pick CUDA when available or CPU otherwise. Set it explicitly when you need to force CPU-only runs.
- Keep speaker channels separate (one WAV per participant) for best turn-merging accuracy.
- Intermediate WAV and TXT caches allow the pipeline to resume without re-transcribing unchanged segments.

---

## Contributing

Issues and pull requests are welcome. Please include reproduction steps for bugs and adhere to existing code style (type hints, tqdm progress bars, etc.).

---

## License & Contact

- License: TODO
````

