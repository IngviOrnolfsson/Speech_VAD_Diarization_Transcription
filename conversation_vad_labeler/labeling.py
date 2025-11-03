import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import entropy

def compute_entropy(text, by='word'):
    text = str(text).strip(); 
    if not text: return 0.0
    tokens = text.split() if by=='word' else list(text)
    if len(tokens)<=1: return 0.0
    counter = Counter(tokens); probs = np.array([v/len(tokens) for v in counter.values()])
    return entropy(probs, base=2)

def classify_transcriptions(df, threshold=1.5):
    df = df.copy(); df['entropy']=df['transcription'].apply(lambda x: compute_entropy(x,'word'))
    df['type']=df['entropy'].apply(lambda x: 'backchannel' if x<threshold else 'turn'); return df

def merge_turns_with_context(df, max_backchannel_dur=1.0, max_gap_sec=2.0):
    df = df.sort_values('start_sec').reset_index(drop=True); merged=[]; i=0
    while i<len(df):
        current = df.iloc[i].copy()
        if current['type']=='backchannel': merged.append(current); i+=1; continue
        speaker = current['speaker']; merged_turn = current.copy(); merged_flag=False; j=i+1
        while j<len(df):
            next_turn = df.iloc[j]
            if next_turn['speaker']!=speaker or next_turn['type']!='turn': break
            gap = next_turn['start_sec']-merged_turn['end_sec']
            if gap>max_gap_sec: break
            between = df[(df['start_sec']<next_turn['start_sec']) & (df['end_sec']>merged_turn['end_sec'])]
            if len(between)==0 or all((between['speaker']==('P1' if speaker=='P2' else 'P2')) & (between['type']=='backchannel') & ((between['end_sec']-between['start_sec'])<=max_backchannel_dur)):
                merged_turn['end_sec']=next_turn['end_sec']
                merged_turn['transcription']=merged_turn['transcription'].strip()+" "+next_turn['transcription'].strip()
                merged_flag=True; j+=1
            else: break
        merged.append(merged_turn); i = j if merged_flag else i+1
    return pd.DataFrame(merged).reset_index(drop=True)
