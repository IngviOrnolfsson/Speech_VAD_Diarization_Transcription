import os
import pandas as pd
from .config import load_config
from .io_helpers import find_audio_files
from .vad import RVADWrapper
from .postprocess_vad import remove_short_segments, filter_low_energy_segments
from .merge_turns import create_turns_df_flexible2
from .transcription import load_whisper_model, transcribe_segments
from .labeling import classify_transcriptions, merge_turns_with_context

def run_pipeline(experiment, trial, cfg_path=None, overwrite=False, device='cpu', language=None):
    cfg = load_config(cfg_path)
    if language:
        cfg['LANGUAGE'] = language

    # Correct paths from config
    audio_root = cfg['paths']['input_audio_dir']
    out_root = os.path.join(cfg['paths']['output_root'], f"EXP{experiment}_T{trial}")
    os.makedirs(out_root, exist_ok=True)

    vad_dir = os.path.join(out_root, cfg['VAD_OUTPUT_SUBDIR'])
    merged_dir = os.path.join(out_root, cfg['MERGED_SUBDIR'])
    segments_dir = os.path.join(out_root, cfg['SEGMENTS_SUBDIR'])
    transcripts_dir = os.path.join(out_root, cfg['TRANSCRIPTS_SUBDIR'])
    for d in [vad_dir, merged_dir, segments_dir, transcripts_dir]:
        os.makedirs(d, exist_ok=True)

    # Now this will search in the correct folder
    p1_wav, p2_wav = find_audio_files(experiment, trial, audio_root)

    vad = RVADWrapper()
    p1_vad = os.path.join(vad_dir, os.path.basename(p1_wav).replace('.wav','.txt'))
    p2_vad = os.path.join(vad_dir, os.path.basename(p2_wav).replace('.wav','.txt'))

    vad.run_vad(p1_wav, p1_vad, min_duration=cfg['VAD_MIN_DURATION'], overwrite=overwrite)
    vad.run_vad(p2_wav, p2_vad, min_duration=cfg['VAD_MIN_DURATION'], overwrite=overwrite)

    p1_df = pd.read_csv(p1_vad, sep='\t', skiprows=1, names=['start','end','label']) 
    p1_df = p1_df[p1_df['label']=='T'].copy(); p1_df['duration']=p1_df['end']-p1_df['start']
    p1_df.reset_index(drop=True,inplace=True)

    p2_df = pd.read_csv(p2_vad, sep='\t', skiprows=1, names=['start','end','label'])
    p2_df = p2_df[p2_df['label']=='T'].copy(); p2_df['duration']=p2_df['end']-p2_df['start']
    p2_df.reset_index(drop=True,inplace=True)

    p1_clean = remove_short_segments(p1_df, cfg['VAD_MIN_DURATION'])
    p2_clean = remove_short_segments(p2_df, cfg['VAD_MIN_DURATION'])
    p1_filt = filter_low_energy_segments(p1_clean, p1_wav, energy_margin_db=cfg['ENERGY_MARGIN_DB'])
    p2_filt = filter_low_energy_segments(p2_clean, p2_wav, energy_margin_db=cfg['ENERGY_MARGIN_DB'])

    p1_filt['speaker']='P1'
    p2_filt['speaker']='P2'

    combined = pd.concat([p1_filt.rename(columns={'start':'start','end':'end','duration':'duration'}), p2_filt.rename(columns={'start':'start','end':'end','duration':'duration'})])
    combined = combined.sort_values(by='start').reset_index(drop=True)

    turns_df = create_turns_df_flexible2(combined, gap_thresh=cfg['GAP_THRESH'], short_utt_thresh=cfg['SHORT_UTT_THRESH'], merge_short_after_long=cfg['MERGE_SHORT_AFTER_LONG'])
    merged_path = os.path.join(merged_dir, f"EXP{experiment}_T{trial}_merged.txt")
    turns_df.to_csv(merged_path, sep='\t', index=False)

    # Prepare segments for transcription
    p1_segments = turns_df[turns_df['Speaker']=='P1'][['Start_Sec','End_Sec','Duration_Sec','Speaker','Turn_Type']].rename(columns={'Start_Sec':'start_sec','End_Sec':'end_sec'})
    p2_segments = turns_df[turns_df['Speaker']=='P2'][['Start_Sec','End_Sec','Duration_Sec','Speaker','Turn_Type']].rename(columns={'Start_Sec':'start_sec','End_Sec':'end_sec'})

    # Transcription
    model = load_whisper_model(device=device, language=cfg['LANGUAGE'])
    p1_results = transcribe_segments(model, p1_segments.reset_index(drop=True), p1_wav, segments_dir, f'EXP{experiment}', trial, 'P1', cache=cfg['CACHE'])
    p2_results = transcribe_segments(model, p2_segments.reset_index(drop=True), p2_wav, segments_dir, f'EXP{experiment}', trial, 'P2', cache=cfg['CACHE'])
    
    df_all = pd.concat([pd.DataFrame(p1_results), pd.DataFrame(p2_results)], ignore_index=True)
    df_class = classify_transcriptions(df_all, threshold=cfg.get('ENTROPY_THRESHOLD',1.5)); classified_path = os.path.join(transcripts_dir, f"EXP{experiment}_T{trial}_combined_classified.txt"); df_class.to_csv(classified_path, sep='\t', index=False)
    df_merged_context = merge_turns_with_context(df_class, max_backchannel_dur=1.0, max_gap_sec=2.0); final_path = os.path.join(transcripts_dir, f"EXP{experiment}_T{trial}_final_labels.txt"); df_merged_context.to_csv(final_path, sep='\t', index=False)

    return {'vad_p1':p1_vad,'vad_p2':p2_vad,'merged_turns':merged_path,'classified':classified_path,'final_labels':final_path}
