import pandas as pd, numpy as np, soundfile as sf
def remove_short_segments(df, min_duration=0.07):
    if df.empty: return df
    return df[(df['end']-df['start'])>=min_duration].reset_index(drop=True)
def compute_rms(audio, sr, start_sec, end_sec):
    start_sample = int(start_sec*sr); end_sample = int(end_sec*sr)
    seg = audio[start_sample:end_sample]; 
    return 0.0 if len(seg)==0 else (seg.astype(float)**2).mean()**0.5
def filter_low_energy_segments(df, audio_path, energy_margin_db=10.0):
    audio, sr = sf.read(audio_path)
    max_rms = (audio.astype(float)**2).mean()**0.5
    max_db = 20 * np.log10(max_rms + 1e-8)
    threshold_db = max_db - energy_margin_db
    valid=[]
    for _,row in df.iterrows():
        rms = compute_rms(audio, sr, row['start'], row['end'])
        rms_db = 20 * np.log10(rms + 1e-8)
        if rms_db >= threshold_db: valid.append(row)
    return pd.DataFrame(valid).reset_index(drop=True)
