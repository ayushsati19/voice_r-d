import os, glob, subprocess
from multiprocessing import Pool
from tqdm import tqdm

SRC = "/home/justdial/change_2/voice_r/dataset_dir/clean_audios"
DST = "/home/justdial/change_2/voice_r/dataset_dir/clean_wav"
SR = 16000
os.makedirs(DST, exist_ok=True)

def convert(src):
    out = os.path.join(DST, os.path.splitext(os.path.basename(src))[0] + ".wav")
    if os.path.exists(out):
        return None
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
         "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", out],
        capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(out): os.remove(out)
        return f"{src}: {r.stderr.strip()[:200]}"
    return None

if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*.mp3"))
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        errors = [e for e in tqdm(pool.imap_unordered(convert, files, chunksize=16),
                                  total=len(files), desc="mp3 -> wav", unit="file") if e]
    print(f"\n{len(files)} files, {len(errors)} errors")
    for e in errors[:10]:
        print("ERR", e)