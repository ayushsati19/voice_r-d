import os, glob, numpy as np, soundfile as sf, pandas as pd
from multiprocessing import Pool
from scipy.signal import resample_poly
from tqdm import tqdm

D = "/home/justdial/change_2/voice_r/dataset_dir"
SRC, DST, SR = f"{D}/noise_audios", f"{D}/noise_wav", 16000
os.makedirs(DST, exist_ok=True)

def convert(p):
    out = os.path.join(DST, os.path.basename(p))
    try:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g)
        sf.write(out, x.astype("float32"), SR, subtype="PCM_16")
        name = os.path.basename(p)[:-4]
        src = "indian" if name.isdigit() else "extra_" + name.rstrip("0123456789_")
        return {"audio_path": out, "duration": len(x) / SR, "source": src}
    except Exception as e:
        return {"audio_path": p, "error": str(e)}

if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*.wav"))
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        rows = list(tqdm(pool.imap_unordered(convert, files, chunksize=64),
                         total=len(files), unit="file"))
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    df = pd.DataFrame(ok)
    df.to_feather(f"{D}/noise_manifest.feather")
    print(f"ok {len(ok)} | errors {len(err)} | {df.duration.sum()/3600:.1f} h")
    print(df.groupby("source").duration.agg(["count", "sum"]).sort_values("sum", ascending=False).head(15))