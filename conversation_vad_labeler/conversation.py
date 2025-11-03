"""High-level conversation processing pipeline utilities."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from .labeling import classify_transcriptions, merge_turns_with_context
from .merge_turns import create_turns_df_windowed
from .postprocess_vad import filter_low_energy_segments
from .transcription import load_whisper_model, transcribe_segments
from .vad import RVADWrapper

EnergyMargin = float | Iterable[float]


def _normalise_margins(margins: EnergyMargin, speakers: List[str]) -> List[float]:
    if isinstance(margins, (list, tuple)):
        if len(margins) != len(speakers):
            raise ValueError("energy_margin_db list length does not match number of speakers")
        return [float(value) for value in margins]
    return [float(margins)] * len(speakers)


def process_conversation(
    speakers_audio: Mapping[str, str],
    output_dir: str = "outputs",
    vad_min_duration: float = 0.07,
    energy_margin_db: EnergyMargin = 10.0,
    gap_thresh: float = 0.2,
    short_utt_thresh: float = 0.7,
    window_sec: float = 2.0,
    merge_short_after_long: bool = True,
    merge_long_after_short: bool = True,
    long_merge_enabled: bool = True,
    merge_max_dur: float = 60.0,
    bridge_short_opponent: bool = True,
    whisper_model_name: str = "large-v3",
    whisper_device: str = "auto",
    whisper_language: str = "da",
    whisper_transformers_batch_size: int = 100,
    entropy_threshold: float = 1.5,
    max_backchannel_dur: float = 1.0,
    max_gap_sec: float = 2.0,
    overwrite: bool = False,
    batch_size: float | None = 60.0,
    interactive_energy_filter: bool = False,
) -> Dict[str, object]:
    """Run the complete VAD→transcription→labeling pipeline for a conversation."""

    speakers = list(speakers_audio.keys())
    if not speakers:
        raise ValueError("speakers_audio must contain at least one entry")

    os.makedirs(output_dir, exist_ok=True)
    energy_margins = _normalise_margins(energy_margin_db, speakers)

    speaker_dirs: Dict[str, str] = {}
    for speaker in speakers:
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        speaker_dirs[speaker] = speaker_dir

    print("Starting conversation processing pipeline...")
    for speaker, path in speakers_audio.items():
        print(f"{speaker} audio: {path}")
    print(f"Output directory: {output_dir}")

    print("\n1. Running Voice Activity Detection...")
    vad = RVADWrapper()
    vad_paths: Dict[str, str] = {}
    for speaker, path in speakers_audio.items():
        vad_path = os.path.join(speaker_dirs[speaker], f"{os.path.splitext(os.path.basename(path))[0]}_vad.txt")
        vad.run_vad(path, vad_path, min_duration=vad_min_duration, overwrite=overwrite)
        vad_paths[speaker] = vad_path
    print("✓ VAD completed")

    print("\n2. Loading and filtering VAD segments...")
    filtered_segments: List[pd.DataFrame] = []
    for idx, speaker in enumerate(speakers):
        audio_path = speakers_audio[speaker]
        vad_path = vad_paths[speaker]

        df = pd.read_csv(vad_path, sep="\t", skiprows=1, names=["start", "end", "label"])
        df = df[df["label"] == "T"].copy()
        df["duration"] = df["end"] - df["start"]
        df.reset_index(drop=True, inplace=True)

        margin_db = energy_margins[idx]
        filt_df = filter_low_energy_segments(
            df,
            audio_path,
            energy_margin_db=margin_db,
            interactive_threshold=interactive_energy_filter,
        )
        filt_df["speaker"] = speaker
        filtered_segments.append(filt_df)

    combined = pd.concat(filtered_segments).sort_values(by="start").reset_index(drop=True)
    speaker_counts = {speaker: int(count) for speaker, count in combined["speaker"].value_counts().items()}
    print(f"✓ Filtered segments: {speaker_counts}")

    print("\n3. Merging turns...")
    merged_turns_path = os.path.join(output_dir, "merged_turns.txt")
    turns_df = create_turns_df_windowed(
        df=combined,
        gap_thresh=gap_thresh,
        short_utt_thresh=short_utt_thresh,
        window_sec=window_sec,
        merge_short_after_long=merge_short_after_long,
        merge_long_after_short=merge_long_after_short,
        long_merge_enabled=long_merge_enabled,
        merge_max_dur=merge_max_dur,
        bridge_short_opponent=bridge_short_opponent,
    )
    turns_df.to_csv(merged_turns_path, sep="\t", index=False)
    print(f"✓ Merged into {len(turns_df)} turns")

    print("\n4. Preparing segments for transcription...")
    segments_by_speaker: Dict[str, pd.DataFrame] = {}
    for speaker in speakers:
        segments_by_speaker[speaker] = turns_df[turns_df["Speaker"] == speaker][
            ["Start_Sec", "End_Sec", "Duration_Sec", "Speaker", "Turn_Type"]
        ].rename(columns={"Start_Sec": "start_sec", "End_Sec": "end_sec"})

    print("\n5. Loading Whisper model and transcribing...")
    model = load_whisper_model(
        model_name=whisper_model_name,
        device=whisper_device,
        language=whisper_language,
        transformers_batch_size=whisper_transformers_batch_size,
    )
    print("✓ Model loaded")

    all_results: List[Dict[str, object]] = []
    for speaker, audio_path in speakers_audio.items():
        print(f"Transcribing {speaker} segments...")
        speaker_segments = segments_by_speaker[speaker]
        results = transcribe_segments(
            model=model,
            segments=speaker_segments.reset_index(drop=True),
            audio_path=audio_path,
            output_dir=speaker_dirs[speaker],
            speaker=speaker,
            cache=True,
            batch_size=batch_size,
            compress=True,
        )
        all_results.extend(results)

    print(f"✓ Transcription completed: {len(all_results)} total segments")

    print("\n6. Classifying transcriptions and merging with context...")
    df_all = pd.DataFrame(all_results)
    df_class = classify_transcriptions(df_all, threshold=entropy_threshold)
    classified_path = os.path.join(output_dir, "classified_transcriptions.txt")
    df_class.to_csv(classified_path, sep="\t", index=False)

    df_merged_context = merge_turns_with_context(
        df_class,
        max_backchannel_dur=max_backchannel_dur,
        max_gap_sec=max_gap_sec,
    )
    final_labels_path = os.path.join(output_dir, "final_labels.txt")
    df_merged_context.to_csv(final_labels_path, sep="\t", index=False)
    print(f"✓ Final processing completed: {len(df_merged_context)} labeled segments")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"Output files saved in: {output_dir}")
    print(f"- VAD results: {list(vad_paths.values())}")
    print(f"- Merged turns: {merged_turns_path}")
    print(f"- Classified transcriptions: {classified_path}")
    print(f"- Final labels: {final_labels_path}")

    return {
        "output_dir": output_dir,
        "vad_paths": vad_paths,
        "merged_turns": merged_turns_path,
        "classified": classified_path,
        "final_labels": final_labels_path,
        "turns_df": turns_df,
        "classified_df": df_class,
        "final_df": df_merged_context,
    }