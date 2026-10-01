import os, glob
import pandas as pd, torch
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm
from silero_vad import get_speech_timestamps

BASE = "/home/justdial/change_2/voice_r"
CLEAN = f"{BASE}/dataset_dir/clean_wav"
MODEL = f"{BASE}/silero_vad.jit"
OUT = f"{BASE}/dataset_dir/clean_labels.feather"
SR = 16000            # label at 16k for accuracy; timestamps are in seconds anyway
WORKERS = max(1, os.cpu_count() - 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    global model
    torch.set_num_threads(1)
    model = torch.jit.load(MODEL).to(DEVICE).eval()

def label(path):
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio).to(DEVICE)
        ts = get_speech_timestamps(
            wav, model, sampling_rate=SR, return_seconds=True,
            threshold=0.6,               # a bit strict: GV audio is noisy
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )
        return {"audio_path": path, "duration": len(wav) / SR,
                "speech_ts": [{"start": float(t["start"]), "end": float(t["end"])} for t in ts]}
    except Exception as e:
        return {"audio_path": path, "error": str(e)}

if __name__ == "__main__":
    files = sorted(f for ext in ("wav", "flac", "mp3", "opus", "m4a")
                   for f in glob.glob(f"{CLEAN}/*.{ext}"))
    with Pool(WORKERS, initializer=init) as pool:
        rows = list(tqdm(pool.imap_unordered(label, files, chunksize=16),
                         total=len(files), desc="Labeling", unit="file"))

    errors = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r and r["speech_ts"]]
    empty = [r for r in rows if "error" not in r and not r["speech_ts"]]

    df = pd.DataFrame(good)
    if not df.empty and "duration" in df.columns:
        df.to_feather(OUT)
        total_audio_h = df["duration"].sum() / 3600
    else:
        total_audio_h = 0.0

    speech_h = sum(t["end"] - t["start"] for r in good for t in r["speech_ts"]) / 3600
    print(f"labeled {len(good)} files | {total_audio_h:.1f} h audio | "
          f"{speech_h:.1f} h speech | no speech: {len(empty)} | errors: {len(errors)}")
    for r in errors[:5]:
        print("ERR", r["audio_path"], r["error"])