import os, shutil
from tqdm import tqdm

DST = "/home/justdial/change_2/voice_r/dataset_dir/noise_audios"
AUDIO = (".wav", ".flac", ".mp3", ".opus")
KEEP_JSON = False          # True = move jsons up too; False = delete them

files = []
for root, _, names in os.walk(DST):
    if root == DST:
        continue
    files += [os.path.join(root, n) for n in names]

moved = skipped = deleted = 0
for src in tqdm(files, desc="Flattening", unit="file"):
    name = os.path.basename(src)
    if name.lower().endswith(".json") and not KEEP_JSON:
        os.remove(src); deleted += 1
        continue
    if not name.lower().endswith(AUDIO + (".json",)):
        continue
    target = os.path.join(DST, name)
    if os.path.exists(target):
        skipped += 1
        continue
    shutil.move(src, target); moved += 1

# remove now-empty folders (deepest first)
for root, dirs, _ in os.walk(DST, topdown=False):
    if root != DST and not os.listdir(root):
        os.rmdir(root)

print(f"moved {moved}, skipped {skipped} (name clash), deleted {deleted} json")