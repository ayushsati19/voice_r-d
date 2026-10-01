"""
Build Silero-VAD tuning data: speech + noise mixtures with frame-accurate labels.

Inputs (from previous steps):
  clean_wav/ + clean_labels.feather     speech (16 kHz) and Silero timestamps
  noise_wav/ + noise_manifest.feather   noise (16 kHz), all non-speech

Outputs:
  mixed/train/*.wav, mixed/val/*.wav    16 kHz mono PCM16 clips
  train.feather, val.feather            audio_path + speech_ts, ready for tune.py
"""
import os, zlib, numpy as np, pandas as pd, soundfile as sf
from multiprocessing import Pool
from scipy.signal import resample_poly
from tqdm import tqdm

D = os.environ.get("DATA_DIR", "/home/justdial/change_2/voice_r/dataset_dir")
OUT = f"{D}/mixed"
SR = 16000

N_CLIPS = {"train": 30000, "val": 3000}   # ~50 h train, ~5 h val at 6 s avg
CLIP_SEC = (4.0, 8.0)
NOISE_ONLY_P = 0.25        # clips with no target speech, speech_ts = []
NO_NOISE_P = 0.05          # speech clips with no added noise
SNR_DB = (0, 20)           # GV speech is already noisy, so no negative SNRs
LEVEL_DBFS = (-35, -15)    # final RMS level, simulates quiet/loud callers
CODEC_P = 0.8              # prob. of 8 kHz + mu-law telephony simulation
INDIAN_SHARE = 0.6         # share of noise draws from the Indian dataset
BOOST = {"extra_AirportAnnouncements": 4.0, "extra_Babble": 2.0,
         "extra_NeighborSpeaking": 2.0}      # oversample these categories
VAL_FRAC = 0.15
SEED = 1234


def crc(s):
    return zlib.crc32(s.encode()) % 1000 / 1000


def load_tables():
    sp = pd.read_feather(f"{D}/clean_labels.feather")
    sp["audio_path"] = (sp.audio_path.str.replace("/clean_audios/", "/clean_wav/", regex=False)
                        .str.replace(r"\.mp3$", ".wav", regex=True))
    # split by the middle id of names like 01-02318-03, so related files stay together
    key = sp.audio_path.map(lambda p: os.path.basename(p).split("-")[1]
                            if os.path.basename(p).count("-") >= 2 else os.path.basename(p))
    sp["split"] = np.where(key.map(crc) < VAL_FRAC, "val", "train")
    sp["speech_ts"] = sp.speech_ts.map(lambda l: [(float(t["start"]), float(t["end"])) for t in l])

    nm = pd.read_feather(f"{D}/noise_manifest.feather")
    ind = nm[nm.source == "indian"].copy()
    ind["n"] = ind.audio_path.map(lambda p: int(os.path.basename(p)[:-4]))
    ind = ind.sort_values("n").reset_index(drop=True)
    # contiguous time split: in every block of 1000 consecutive chunks, the last 15% go to val
    ind["split"] = np.where((ind.n % 1000) >= 1000 * (1 - VAL_FRAC), "val", "train")
    indian = {s: ind[ind.split == s].audio_path.tolist() for s in ("train", "val")}

    ext = nm[nm.source != "indian"]
    extras = {src: g[["audio_path", "duration"]].values.tolist() for src, g in ext.groupby("source")}

    speech = {s: sp[sp.split == s][["audio_path", "duration", "speech_ts"]].values.tolist()
              for s in ("train", "val")}
    return speech, indian, extras


def init(speech, indian, extras):
    global SPEECH, INDIAN, EXTRAS, EXTRA_SRC, EXTRA_W
    SPEECH, INDIAN, EXTRAS = speech, indian, extras
    EXTRA_SRC = sorted(extras)
    w = np.array([BOOST.get(s, 1.0) for s in EXTRA_SRC])
    EXTRA_W = w / w.sum()


def read(path, start=0, stop=None):
    x, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


def crossfade_concat(parts, fade=80):
    out = parts[0]
    for p in parts[1:]:
        if len(out) > fade and len(p) > fade:
            r = np.linspace(0, 1, fade, dtype="float32")
            out[-fade:] = out[-fade:] * (1 - r) + p[:fade] * r
            p = p[fade:]
        out = np.concatenate([out, p])
    return out


def indian_noise(rng, n, split):
    files = INDIAN[split]
    i, parts, total, fade = int(rng.integers(len(files))), [], 0, 80
    while total < n:                          # consecutive chunks = continuous audio
        x = read(files[i % len(files)])
        parts.append(x); total += len(x) - (fade if len(parts) > 1 else 0); i += 1
    out = crossfade_concat(parts, fade)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)), mode="wrap")
    return out[:n]


def extra_noise(rng, n, split):
    src = EXTRA_SRC[rng.choice(len(EXTRA_SRC), p=EXTRA_W)]
    path, dur = EXTRAS[src][rng.integers(len(EXTRAS[src]))]
    total = int(dur * SR)
    lo, hi = (0, int(total * (1 - VAL_FRAC))) if split == "train" else (int(total * (1 - VAL_FRAC)), total)
    if hi - lo >= n:
        s = int(rng.integers(lo, hi - n + 1))
        return read(path, s, s + n)
    x = read(path, lo, hi)
    return np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n]


def get_noise(rng, n, split):
    if not EXTRA_SRC or rng.random() < INDIAN_SHARE:
        return indian_noise(rng, n, split)
    return extra_noise(rng, n, split)


def telephony(y):
    x = resample_poly(y, 1, 2)                         # 16k -> 8k band-limit
    x = np.clip(x, -1, 1)
    mu = 255.0
    c = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1        # 8-bit quantization
    x = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    return resample_poly(x, 2, 1).astype("float32")    # back to 16k for tune.py


def speech_clip(rng, n, split):
    rows = SPEECH[split]
    for _ in range(10):
        path, dur, ts = rows[rng.integers(len(rows))]
        if not ts:
            continue
        lead = rng.uniform(0.2, 1.5)
        tail = rng.uniform(0.2, 1.0)
        seg = n / SR - lead - tail
        if seg < 1.0:
            continue
        a, _ = ts[rng.integers(len(ts))]
        s0 = float(np.clip(a - rng.uniform(0, 0.3 * seg), 0, max(0.0, dur - seg)))
        x = read(path, int(s0 * SR), int((s0 + seg) * SR))
        buf = np.zeros(n, dtype="float32")
        o = int(lead * SR)
        buf[o:o + len(x)] = x[: n - o]
        labels = []
        for st, en in ts:
            st2, en2 = max(st, s0) - s0 + lead, min(en, s0 + len(x) / SR) - s0 + lead
            if en2 - st2 >= 0.1:
                labels.append({"start": round(st2, 3), "end": round(en2, 3)})
        if labels:
            return buf, labels
    return None, None


def make(args):
    idx, split = args
    rng = np.random.default_rng(SEED + idx + (10_000_000 if split == "val" else 0))
    n = int(rng.uniform(*CLIP_SEC) * SR)

    if rng.random() < NOISE_ONLY_P:
        y, labels = get_noise(rng, n, split), []
    else:
        y, labels = speech_clip(rng, n, split)
        if y is None:
            y, labels = get_noise(rng, n, split), []
        elif rng.random() >= NO_NOISE_P:
            mask = np.zeros(n, bool)
            for t in labels:
                mask[int(t["start"] * SR):int(t["end"] * SR)] = True
            ps = np.mean(y[mask] ** 2) + 1e-10
            nz = get_noise(rng, n, split)
            pn = np.mean(nz ** 2) + 1e-10
            snr = rng.uniform(*SNR_DB)
            y = y + nz * np.sqrt(ps / (pn * 10 ** (snr / 10)))

    rms = np.sqrt(np.mean(y ** 2)) + 1e-10
    y = y * (10 ** (rng.uniform(*LEVEL_DBFS) / 20) / rms)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    if rng.random() < CODEC_P:
        y = telephony(y)

    out = f"{OUT}/{split}/{idx:06d}.wav"
    sf.write(out, y, SR, subtype="PCM_16")
    return {"audio_path": out, "speech_ts": labels}


if __name__ == "__main__":
    speech, indian, extras = load_tables()
    print("speech files:", {s: len(v) for s, v in speech.items()},
          "| indian noise chunks:", {s: len(v) for s, v in indian.items()},
          "| extra categories:", len(extras))
    workers = max(1, os.cpu_count() - 2)
    for split, n in N_CLIPS.items():
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        with Pool(workers, initializer=init, initargs=(speech, indian, extras)) as pool:
            rows = list(tqdm(pool.imap_unordered(make, [(i, split) for i in range(n)], chunksize=32),
                             total=n, desc=split, unit="clip"))
        rows.sort(key=lambda r: r["audio_path"])
        df = pd.DataFrame(rows)
        df.to_feather(f"{D}/{split}.feather")
        n_sp = int((df.speech_ts.map(len) > 0).sum())
        print(f"{split}: {len(df)} clips | with speech {n_sp} | noise-only {len(df) - n_sp} -> {D}/{split}.feather")