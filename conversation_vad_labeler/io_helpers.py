from pathlib import Path
import re

def find_audio_files(experiment, trial, audio_root):
    """Locate p1/p2 audio files for a given experiment and trial with debug info."""
    print(f"[DEBUG] Searching for EXP{experiment} trial{trial} in {audio_root}")
    audio_root = Path(audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"[DEBUG] Audio root does not exist: {audio_root.resolve()}")

    exp_str = f"EXP{experiment}_".lower()
    trial_str = f"_trial{trial}".lower()
    p1, p2 = None, None

    # Loop through files and match both experiment and trial
    for p in audio_root.rglob('*.wav'):
        name = p.name.lower()
        if exp_str in name and "_p1" in name and trial_str in name:
            p1 = p
        if exp_str in name and "_p2" in name and trial_str in name:
            p2 = p

    print(f"[DEBUG] Found p1: {p1}")
    print(f"[DEBUG] Found p2: {p2}")

    # Debug: show all EXP{experiment} files
    print("[DEBUG] All matching EXP files in directory:")
    for f in audio_root.rglob(f"EXP{experiment}_*.wav"):
        print(f"   {f.name}")

    if not p1 or not p2:
        raise FileNotFoundError(f"Could not find both p1/p2 files for EXP{experiment} trial{trial} in {audio_root}")

    return str(p1), str(p2)
