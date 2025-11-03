import os
import pandas as pd
from conversation_vad_labeler.pipeline import run_pipeline
# # This test assumes the two sample wav files are in the current directory and named:
# # EXP0_None_p1_trial0.wav and EXP0_None_p2_trial0.wav
# print('Running quick test (EXP0 T0)')
# out = run_pipeline(0, 0, cfg_path='config.yaml', overwrite=False, device='cpu', language='da')
# print('Test outputs:\n', out)

config_path = "/Users/hahea/Documents/Experiment_12/Analysis/Audio/conversation_vad_labeler_package/config.yaml"

outputs = run_pipeline(
    experiment=9,
    trial=2,
    cfg_path=config_path,
    overwrite=False,
    device="cpu",
    language="da"
)

print("\n=== Test Completed ===")
print("Generated outputs:\n", outputs)

