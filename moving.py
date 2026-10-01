import os, shutil
from tqdm import tqdm

SRC = "GV_Train_100h/Audio"      # <- adjust path
DST = "/home/justdial/change_2/voice_r/dataset_dir/clean_audios"
AUDIO = (".wav", ".flac", ".mp3", ".opus", ".m4a")

os.makedirs(DST, exist_ok=True)
files = [os.path.join(r, n) for r, _, ns in os.walk(SRC) for n in ns
         if n.lower().endswith(AUDIO)]

moved = skipped = 0
for src in tqdm(files, desc="Moving clean", unit="file"):
    target = os.path.join(DST, os.path.basename(src))
    if os.path.exists(target):
        skipped += 1
        continue
    shutil.move(src, target); moved += 1

print(f"found {len(files)}, moved {moved}, skipped {skipped} (name clash)")