import os
import torchaudio
from rVADfast import rVADfast

def convert_to_labels(
    vad_timestamps: list[float],
    vad_labels: list[int]
) -> list[tuple[float, float]]:
    """
    Convert VAD timestamps and labels into speech intervals.

    Args:
        vad_timestamps: List of timestamps corresponding to VAD labels.
        vad_labels: List of binary labels (0 or 1) indicating speech.

    Returns:
        List of tuples (start_time, end_time) for speech intervals.
    """
    speech_intervals = []
    talking = False
    start_time = 0.0

    for i, label in enumerate(vad_labels):
        if label == 1 and not talking:
            start_time = vad_timestamps[i]
            talking = True
        elif label == 0 and talking:
            end_time = vad_timestamps[i]
            speech_intervals.append((start_time, end_time))
            talking = False

    # Handle case where speech continues to the end
    if talking:
        speech_intervals.append((start_time, vad_timestamps[-1]))

    return speech_intervals

class RVADWrapper:
    """
    Wrapper class for rVADfast Voice Activity Detection.

    Provides a simple interface to run VAD on audio files and save results.
    """

    def __init__(self) -> None:
        """Initialize the VAD instance."""
        self.vad = rVADfast()

    def run_vad(
        self,
        wav_path: str,
        out_txt_path: str,
        min_duration: float = 0.07,
        overwrite: bool = False
    ) -> str:
        """
        Run Voice Activity Detection on a WAV file and save intervals to text file.

        Args:
            wav_path: Path to the input WAV file.
            out_txt_path: Path to the output text file for VAD intervals.
            min_duration: Minimum duration for speech segments to be included.
            overwrite: Whether to overwrite existing output file.

        Returns:
            Path to the output text file.
        """
        if os.path.exists(out_txt_path) and not overwrite:
            return out_txt_path

        # Load audio
        signal, fs = torchaudio.load(wav_path)
        signal_np = signal.numpy().flatten()

        # Run VAD
        vad_labels, vad_timestamps = self.vad(signal_np, fs)

        # Convert to intervals
        intervals = convert_to_labels(vad_timestamps, vad_labels)

        # Write to file, filtering by min_duration
        with open(out_txt_path, 'w') as f:
            f.write('Start_Time(s)\tEnd_Time(s)\tAnnotation\n')
            for start, end in intervals:
                if (end - start) >= min_duration:
                    f.write(f"{start:.2f}\t{end:.2f}\tT\n")

        return out_txt_path
