import os, soundfile as sf
import whisperx, numpy as np
def load_whisper_model(model_name='large-v3', device='cpu', language='da', compute_type='int8', cache_dir=None):
    # whisperx will handle caching; user can set XDG_CACHE_HOME to change location
    model = whisperx.load_model(model_name, device=device, language=language, compute_type=compute_type)
    return model
def _save_segment_wav(out_path, audio_array, sr=16000):
    sf.write(out_path, audio_array, samplerate=sr, subtype='PCM_16')
def transcribe_segments(model, segments, audio_path, output_dir, exp_id, trial_id, speaker, cache=True):
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = sf.read(audio_path); results=[]
    for idx, seg in segments.iterrows():
        start, end = seg['start_sec'], seg['end_sec']
        seg_filename = os.path.join(output_dir, f"{exp_id}_T{trial_id}_{speaker}_seg_{idx}_{start:.2f}_{end:.2f}.wav")
        txt_cache = seg_filename.replace('.wav', '.txt')
        start_samp, end_samp = int(start*sr), int(end*sr)
        segment_audio = audio[start_samp:end_samp]
        if (end-start)*sr < 1600:
            results.append({'speaker':speaker,'start_sec':start,'end_sec':end,'transcription':''}); continue
        if cache and os.path.exists(txt_cache):
            with open(txt_cache,'r') as f: text=f.read().strip()
            results.append({'speaker':speaker,'start_sec':start,'end_sec':end,'transcription':text}); continue
        _save_segment_wav(seg_filename, segment_audio, sr=sr)
        try:
            res = model.transcribe(seg_filename)
            text = ' '.join([s['text'].strip() for s in res.get('segments',[])])
        except Exception as e:
            text = f"ERROR: {e}"
        with open(txt_cache,'w') as f: f.write(text)
        results.append({'speaker':speaker,'start_sec':start,'end_sec':end,'transcription':text})
    return results
