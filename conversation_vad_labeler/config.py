import os, yaml
DEFAULTS = {
    'AUDIO_ROOT': './',            # override in config.yaml or CLI
    'OUTPUT_ROOT': './outputs',
    'VAD_OUTPUT_SUBDIR': 'vad',
    'MERGED_SUBDIR': 'merged_turns',
    'SEGMENTS_SUBDIR': 'segments',
    'TRANSCRIPTS_SUBDIR': 'transcriptions',
    'LANGUAGE': 'da',
    'VAD_MIN_DURATION': 0.07,
    'ENERGY_MARGIN_DB': 10.0,
    'GAP_THRESH': 0.2,
    'SHORT_UTT_THRESH': 0.7,
    'MERGE_SHORT_AFTER_LONG': True,
    'CACHE': True,
    'ENTROPY_THRESHOLD': 1.5
}
def load_config(cfg_path=None):
    cfg = DEFAULTS.copy()
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            user_cfg = yaml.safe_load(f)
            if user_cfg: cfg.update(user_cfg)
    return cfg
