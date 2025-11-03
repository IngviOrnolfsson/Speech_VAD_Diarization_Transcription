import pandas as pd
import numpy as np

def create_turns_df_flexible2(
    df: pd.DataFrame,
    gap_thresh: float = 0.2,
    short_utt_thresh: float = 0.7,
    merge_short_after_long: bool = True,
) -> pd.DataFrame:
    """
    Merge VAD segments into speaker turns using a flexible merging strategy.

    This function iterates through segments and merges consecutive segments
    from the same speaker based on gap thresholds and utterance lengths.
    It handles merging short utterances after long ones, and bridging
    short opponent utterances.

    Args:
        df: DataFrame with columns 'speaker', 'start', 'end', 'duration'.
        gap_thresh: Maximum gap (in seconds) to merge segments from same speaker.
        short_utt_thresh: Threshold (in seconds) to classify utterances as short.
        merge_short_after_long: Whether to merge short utterances after long ones.

    Returns:
        DataFrame with merged turns, columns: Speaker, Start_Sec, End_Sec, Duration_Sec, Turn_Type.
    """
    turns = []
    n = len(df)
    i = 0

    while i < n:
        current = df.iloc[i].copy()
        j = i + 1

        while j < n:
            next_seg = df.iloc[j]
            gap = next_seg['start'] - current['end']

            if next_seg['speaker'] == current['speaker']:
                # Merge if gap is small
                if gap < gap_thresh:
                    current['end'] = max(current['end'], next_seg['end'])
                    current['duration'] = current['end'] - current['start']
                    j += 1
                    continue
                # Merge if both are long enough
                elif (current['duration'] >= short_utt_thresh and
                      next_seg['duration'] >= short_utt_thresh):
                    current['end'] = max(current['end'], next_seg['end'])
                    current['duration'] = current['end'] - current['start']
                    j += 1
                    continue
                # Merge short after long
                elif (merge_short_after_long and
                      current['duration'] >= short_utt_thresh and
                      next_seg['duration'] < short_utt_thresh):
                    current['end'] = max(current['end'], next_seg['end'])
                    current['duration'] = current['end'] - current['start']
                    j += 1
                    continue
                else:
                    break
            else:
                # Bridge short opponent utterance if next is same speaker
                if (next_seg['duration'] < short_utt_thresh and
                    j + 1 < n and
                    df.iloc[j + 1]['speaker'] == current['speaker']):
                    combined_duration = current['duration'] + df.iloc[j + 1]['duration']
                    if combined_duration >= short_utt_thresh:
                        current['end'] = df.iloc[j + 1]['end']
                        current['duration'] = current['end'] - current['start']
                        j += 2
                        continue
                break

        turns.append({
            'Speaker': current['speaker'],
            'Start_Sec': current['start'],
            'End_Sec': current['end'],
            'Duration_Sec': current['duration'],
            'Turn_Type': 'T'
        })
        i = j

    return pd.DataFrame(turns)



def _extend_segment(segment: pd.Series, new_end: float) -> None:
    """
    Update a segment to end at new_end and refresh its duration.

    Args:
        segment: The segment series to update (modified in-place).
        new_end: The new end time for the segment.
    """
    segment["end"] = max(segment["end"], new_end)
    segment["duration"] = segment["end"] - segment["start"]

def create_turns_df_windowed(
    df: pd.DataFrame,
    gap_thresh: float = 0.2,
    short_utt_thresh: float = 0.7,
    window_sec: float = 2.0,
    merge_short_after_long: bool = True,
    merge_long_after_short: bool = True,
    long_merge_enabled: bool = True,
    # long_merge_min_dur: float | None = None,
    merge_max_dur: float = 60.0,
    bridge_short_opponent: bool = True,
) -> pd.DataFrame:
    """Merge VAD segments into speaker turns by scanning forward within a time window."""
    if df.empty:
        return pd.DataFrame(columns=["Speaker", "Start_Sec", "End_Sec", "Duration_Sec", "Turn_Type"])

    # Sort segments by start time and ensure duration column exists
    segments = df.sort_values("start").reset_index(drop=True).copy()

    if "duration" not in segments.columns:
        segments["duration"] = segments["end"] - segments["start"]

    # Set default for merge_max_dur if not provided
    if merge_max_dur is None:
        merge_max_dur = np.inf

    turns: list[dict[str, float | str]] = []
    n = len(segments)
    i = 0

    while i < n:
        current = segments.iloc[i].copy()
        speaker = current["speaker"]
        j = i + 1

        while True:
            # Find segments within the time window
            window_mask = (segments.index >= j) & (segments["start"] <= current["end"] + window_sec)
            window = segments.loc[window_mask]
            if window.empty:
                break

            candidate = window.iloc[0]
            candidate_idx = candidate.name
            gap_from_current = max(0.0, candidate["start"] - current["end"])

            if candidate["speaker"] == speaker:
                # Merge if gap is within threshold
                if gap_from_current <= gap_thresh:
                    _extend_segment(current, candidate["end"])
                    j = candidate_idx + 1
                    continue

                # Merge short utterance after long one
                if (
                    merge_short_after_long
                    and current["duration"] >= short_utt_thresh
                    and candidate["duration"] < short_utt_thresh
                    and current['duration'] + candidate['duration'] >= gap_from_current
                    and current['duration'] + candidate['duration'] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    j = candidate_idx + 1
                    continue

                # Merge long utterance after short one
                if (
                    merge_long_after_short
                    and current["duration"] < short_utt_thresh
                    and candidate["duration"] >= short_utt_thresh
                    and current['duration'] + candidate['duration'] >= gap_from_current
                    and current['duration'] + candidate['duration'] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    j = candidate_idx + 1
                    continue

                # Merge two long utterances
                if (
                    long_merge_enabled
                    and current["duration"] >= short_utt_thresh
                    and candidate["duration"] >= short_utt_thresh
                    and current['duration'] + candidate['duration'] >= gap_from_current
                    and current['duration'] + candidate['duration'] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    j = candidate_idx + 1
                    continue
                break

            # Bridge short opponent utterances if enabled
            if not bridge_short_opponent:
                break

            # Look for same speaker segments within short utterance threshold
            sub_window_mask = (segments.index >= j) & (
                segments["start"] <= current["end"] + short_utt_thresh
            )
            sub_window = segments.loc[sub_window_mask]

            if sub_window.empty:
                break

            same_speaker_rows = sub_window[sub_window["speaker"] == speaker]
            if same_speaker_rows.empty:
                break

            target = same_speaker_rows.iloc[-1]
            _extend_segment(current, target["end"])
            j = target.name + 1
            continue

        # Add the completed turn to the list
        turns.append(
            {
                "Speaker": speaker,
                "Start_Sec": current["start"],
                "End_Sec": current["end"],
                "Duration_Sec": current["duration"],
                "Turn_Type": "T",
            }
        )
        i = j

    return pd.DataFrame(turns)


