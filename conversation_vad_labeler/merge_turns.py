import pandas as pd

def create_turns_df_flexible2(df, gap_thresh=0.2, short_utt_thresh=0.7, merge_short_after_long=True):
    turns=[]; n=len(df); i=0
    while i<n:
        current = df.iloc[i].copy(); j=i+1
        while j<n:
            next_seg = df.iloc[j]; gap = next_seg['start']-current['end']
            if next_seg['speaker']==current['speaker']:
                if gap < gap_thresh:
                    current['end']=max(current['end'], next_seg['end']); current['duration']=current['end']-current['start']; j+=1; continue
                elif current['duration']>=short_utt_thresh and next_seg['duration']>=short_utt_thresh:
                    current['end']=max(current['end'], next_seg['end']); current['duration']=current['end']-current['start']; j+=1; continue
                elif merge_short_after_long and current['duration']>=short_utt_thresh and next_seg['duration']<short_utt_thresh:
                    current['end']=max(current['end'], next_seg['end']); current['duration']=current['end']-current['start']; j+=1; continue
                else: break
            else:
                if next_seg['duration']<short_utt_thresh and j+1<n and df.iloc[j+1]['speaker']==current['speaker']:
                    combined_duration = current['duration'] + df.iloc[j+1]['duration']
                    if combined_duration >= short_utt_thresh:
                        current['end'] = df.iloc[j+1]['end']; current['duration']=current['end']-current['start']; j+=2; continue
                break
        turns.append({'Speaker':current['speaker'],'Start_Sec':current['start'],'End_Sec':current['end'],'Duration_Sec':current['duration'],'Turn_Type':'T'})
        i=j
    return pd.DataFrame(turns)
