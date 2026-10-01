import json, glob, os, re, shutil
from tqdm import tqdm

SRC = "/home/justdial/change_2/voice_r/data/indian-noise-dataset/extracted"
DST = "/home/justdial/change_2/voice_r/dataset_dir/noise_audios"

jsons = glob.glob(f"{SRC}/**/*.json", recursive=True)
moved = skipped = 0

for jp in tqdm(jsons, desc="Moving", unit="clip"):
    ntype = json.load(open(jp))["noise_type"]
    folder = os.path.join(DST, re.sub(r"[^A-Za-z0-9]+", "_", ntype).strip("_"))
    os.makedirs(folder, exist_ok=True)

    for f in glob.glob(jp[:-5] + ".*"):          # audio file + the json itself
        target = os.path.join(folder, os.path.basename(f))
        if os.path.exists(target):
            skipped += 1
            continue
        shutil.move(f, target)
        moved += 1

print(f"moved {moved} files, skipped {skipped} (already existed)")