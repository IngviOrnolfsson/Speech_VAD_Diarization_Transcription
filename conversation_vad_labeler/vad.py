import os, torchaudio
import numpy as np
from rVADfast import rVADfast
def convert_to_labels(vad_timestamps, vad_labels):
    speech_intervals = []
    talking = False
    start_time = 0
    for i, label in enumerate(vad_labels):
        if label == 1 and not talking:
            start_time = vad_timestamps[i]; talking = True
        elif label == 0 and talking:
            end_time = vad_timestamps[i]; speech_intervals.append((start_time, end_time)); talking = False
    if talking: speech_intervals.append((start_time, vad_timestamps[-1]))
    return speech_intervals
class RVADWrapper:
    def __init__(self): self.vad = rVADfast()
    def run_vad(self, wav_path, out_txt_path, min_duration=0.07, overwrite=False):
        if os.path.exists(out_txt_path) and not overwrite: return out_txt_path
        signal, fs = torchaudio.load(wav_path); signal_np = signal.numpy().flatten()
        vad_labels, vad_timestamps = self.vad(signal_np, fs)
        intervals = convert_to_labels(vad_timestamps, vad_labels)
        with open(out_txt_path, 'w') as f:
            f.write('Start_Time(s)\tEnd_Time(s)\tAnnotation\n')
            for start,end in intervals:
                if (end-start) >= min_duration: f.write(f"{start:.2f}\t{end:.2f}\tT\n")
        return out_txt_path
