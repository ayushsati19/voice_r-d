import pandas as pd, soundfile as sf
df = pd.read_feather("/home/justdial/change_2/voice_r/dataset_dir/val.feather")
tot = sum(sf.info(p).duration for p in df.audio_path)
sp = sum(t["end"] - t["start"] for l in df.speech_ts for t in l)
print(f"speech frames: {sp/tot:.1%}  -> always-'no speech' accuracy = {1-sp/tot:.1%}")