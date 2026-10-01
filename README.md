# Silero VAD fine-tuning for noisy telephony

Fine-tune Silero VAD so it fires on the caller's speech but ignores background noise and background voices: markets, railway and airport announcements, cafeterias, traffic, people talking nearby.

The repo contains scripts only. Data, generated datasets and model files are not included; the steps below create them.

Result from the reference run: validation ROC-AUC 0.885 for the tuned 16 kHz head, exported to ONNX with a JIT-vs-ONNX max difference under 1e-6.

## How it works

1. Clean phone speech is labeled automatically with stock Silero (on clean audio its timestamps are reliable).
2. Noise is never labeled by the VAD. All noise, including background speech, has the label "no speech".
3. Speech and noise are mixed into short clips at random SNRs, levels and with a telephony codec simulation. Labels carry over from the clean speech, so no hand labeling is needed.
4. Silero's own tuning scripts fine-tune the 16 kHz head on those clips.
5. The tuned head is exported to ONNX for production.

## Requirements

- Linux, Python 3.10, ffmpeg
- NVIDIA GPU recommended (an 8 GB card is plenty; CPU works but is slow)
- About 40 GB free disk (WAV speech ~11 GB, noise, ~6.5 GB of mixtures, plus originals)

```bash
python -m venv r_venv && source r_venv/bin/activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install silero-vad soundfile scipy numpy pandas pyarrow tqdm huggingface_hub[cli,hf_transfer] \
            omegaconf scikit-learn audiomentations onnx onnxruntime
sudo apt install -y ffmpeg
```

Install torch and torchaudio as a matched pair. A mismatched torchaudio fails to import with errors like `libcudart.so.13: cannot open shared object file`. The cu124 build runs on any driver that supports CUDA 12.4 or newer.

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Scripts

| Script | Step | Output |
| --- | --- | --- |
| `hugging_face_download.py` | Download the Indian noise dataset | `data/indian-noise-dataset/train/*.tar` |
| `conversion_to_wav.py` | Convert clean speech MP3 to 16 kHz WAV | `dataset_dir/clean_wav/` |
| `label_clean.py` | Label clean speech with stock Silero | `dataset_dir/clean_labels.feather` |
| `prepare_noise.py` | Resample noise to 16 kHz and build a manifest | `dataset_dir/noise_wav/`, `noise_manifest.feather` |
| `make_mixtures.py` | Generate labeled training clips | `dataset_dir/mixed/`, `train.feather`, `val.feather` |
| `convert_to_onnx.py` | Export the tuned 16k head and verify it | `silero_vad_tuned_16k.onnx` |

Every script has its paths as constants at the top (`BASE`, `D`, `SRC`, `DST`). Edit them to match your machine before running.

## Directory layout after setup

```
voice_r/
├── silero_vad.jit                 # stock model (download, see step 0)
├── silero-vad/                    # cloned Silero repo, provides tuning/
├── data/indian-noise-dataset/     # noise download + extracted tars
└── dataset_dir/
    ├── clean_audios/              # original speech (MP3)
    ├── clean_wav/                 # speech, 16 kHz mono PCM16
    ├── clean_labels.feather
    ├── noise_audios/              # all noise, one flat folder
    ├── noise_wav/                 # noise, 16 kHz mono PCM16
    ├── noise_manifest.feather
    ├── mixed/train/, mixed/val/
    ├── train.feather, val.feather
```

Create the folders:

```bash
mkdir -p dataset_dir/{clean_audios,noise_audios} data
```

## Step 0: Get the code and the base model

```bash
git clone https://github.com/snakers4/silero-vad.git
```

Use a Silero JIT model as the starting point and save it as `silero_vad.jit`. Inspect its interface before going further:

```python
import torch
m = torch.jit.load("silero_vad.jit", map_location="cpu")
print(m.code)          # top-level forward(x, sr)
print(m._model.code)   # 16k head: forward(x, h, c) -> (out, h, c)
```

The scripts here assume the architecture with separate LSTM `h` and `c` of shape `[2, batch, 64]`. A newer model with a single `state` input needs changes to the export step.

## Step 1: Get the data

**Clean speech.** Phone-quality speech that sounds like real callers. The reference run used `GV_Train_100h` (37,152 MP3 files, about 100 h). Copy or move the audio files, flat, into `dataset_dir/clean_audios/`.

**Noise.** Download the Indian noise dataset:

```bash
python hugging_face_download.py
# or:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ajajali09/indian-noise-dataset \
  --repo-type dataset --local-dir ./data/indian-noise-dataset
```

The download is 97 WebDataset tars (`00000.tar` to `00096.tar`). Extract them:

```bash
cd data/indian-noise-dataset/train && mkdir -p ../extracted
ls *.tar | xargs -P 8 -I{} sh -c 'n=$(basename {} .tar); mkdir -p ../extracted/$n && tar -xf {} -C ../extracted/$n'
```

Each chunk is a short (~0.86 s) 48 kHz WAV with a JSON holding its `noise_type`. Move all audio into `dataset_dir/noise_audios/` as one flat folder. The JSONs are optional after this, since every noise file gets the same label.

Add any extra noise (airport announcements, babble, office, appliances) to the same folder. Keep airport and other rare categories, because the mixer oversamples them by filename prefix.

## Step 2: Convert clean speech to WAV

```bash
python conversion_to_wav.py
```

Converts every MP3 to 16 kHz mono 16-bit WAV in `clean_wav/` using ffmpeg. It skips files that already exist, so it is safe to re-run.

Verify:

```bash
ls dataset_dir/clean_wav | wc -l                       # same count as the MP3s
du -sh dataset_dir/clean_wav                           # ~115 MB per hour of audio
find dataset_dir/clean_wav -name "*.wav" -size -1k | wc -l   # should be 0
```

A run that finishes at hundreds of thousands of files per second did nothing: every output already existed.

## Step 3: Label the clean speech

```bash
python label_clean.py
```

Runs stock Silero at 16 kHz on every clean file (threshold 0.6, minimum speech 250 ms, minimum silence 100 ms, padding 30 ms) and writes `clean_labels.feather` with columns `audio_path`, `duration`, `speech_ts`. Files with no detected speech are dropped.

Load audio with soundfile, not `silero_vad.read_audio`, which needs an extra audio backend. Expect a normal speed (tens to hundreds of files per second). If the progress bar flies through instantly, every file failed; read the printed errors.

Spot-check 20 files in Audacity. Every training label comes from these timestamps.

## Step 4: Prepare the noise

```bash
python prepare_noise.py
```

Resamples all noise to 16 kHz mono PCM16 in `noise_wav/` and writes `noise_manifest.feather` with `audio_path`, `duration` and `source` (`indian` for numeric filenames, `extra_<Category>` for named ones).

Never run the VAD on noise. It would label background voices as speech, which is the behavior this project trains out.

## Step 5: Generate the training clips

```bash
python make_mixtures.py
```

Writes 30,000 train clips (~50 h) and 3,000 val clips (~5 h) at 16 kHz, plus `train.feather` and `val.feather`.

What the mixer does:

- 75% speech clips: 4–8 s windows around real speech with 0.2–1.5 s lead-in and 0.2–1.0 s tail, noise added at 0–20 dB SNR (measured on speech frames).
- 25% noise-only clips with an empty label list.
- Final level randomized to −35 to −15 dBFS.
- 80% of clips go through 8 kHz downsampling, 8-bit μ-law and back to 16 kHz, to sound like a phone line.
- Noise: 60% from the Indian set (consecutive chunks joined with short crossfades), 40% from extra categories, with airport announcements ×4 and babble and neighbor speaking ×2.
- Train and val share no speakers or noise segments.

All of these are constants at the top of the script. Listen to about 10 clips, including a few noise-only ones, before training.

## Step 6: Fine-tune

Edit `silero-vad/tuning/config.yml`. Change values only; every key must stay, or `tune.py` fails with a missing-key error (`git checkout config.yml` restores the original).

```yaml
jit_model_path: /path/to/silero_vad.jit
use_torchhub: False
tune_8k: False
train_dataset_path: /path/to/dataset_dir/train.feather
val_dataset_path: /path/to/dataset_dir/val.feather
model_save_path: /path/to/silero_vad_tuned_16k.jit
noise_loss: 1.0
max_train_length_sec: 8
aug_prob: 0.2
learning_rate: 5e-4
batch_size: 128
num_workers: 8
num_epochs: 20
device: cuda
```

`tune_8k: False` is required. The clips are 16 kHz, and feeding them to the 8k head trains nothing: ROC-AUC stays flat around 0.59.

Get a baseline, then train, inside `screen` or `tmux`:

```bash
cd silero-vad/tuning
python search_thresholds.py                 # stock model baseline
python tune.py 2>&1 | tee tune.log
```

Expected: ROC-AUC rising across epochs and train loss falling (reference run: 0.885, train 0.43, val 0.37). Val loss below train loss is normal, because only train clips get extra augmentation. If ROC-AUC is flat from epoch 1, stop and check the sample rate, `tune_8k` and the dataset paths.

Then point `jit_model_path` at the tuned model and search thresholds again:

```bash
python search_thresholds.py
```

Use these thresholds in production, not 0.5.

## Step 7: Export to ONNX

```bash
python convert_to_onnx.py
```

Exports `m._model` (the 16k head) directly, with inputs `input`, `h`, `c` and outputs `output`, `hn`, `cn`, then runs a val clip through both JIT and ONNX chunk by chunk. Deploy only if it prints `OK` (max difference under 1e-4).

The tuned JIT is saved from GPU; load it with `map_location="cpu"` on CPU machines.

## Using the model in production

- Upsample 8 kHz SIP audio to 16 kHz before the VAD.
- Feed 512-sample chunks (32 ms). Start with `h` and `c` as zeros `[2, 1, 64]` and pass `hn`/`cn` back in on every next chunk.
- Use the thresholds from `search_thresholds.py` on the tuned model.
- The tuned ONNX has no `sr` input and contains only the 16k head. If your calling code passes `sr`, either remove it or add a dummy input, and make sure the audio really is 16 kHz.

## Evaluating properly

The val set chose both the checkpoint and the thresholds, so its scores are optimistic. Before shipping:

1. Generate a synthetic test set from the val-side sources with a different seed and read accuracy only.
2. Hand-label a few hundred real noisy calls (pre-label with stock Silero, correct in Audacity or Label Studio).
3. Compare stock vs tuned on false triggers, missed speech and onset/offset delay, per noise type.

If the tuned model still fires on announcements, raise `noise_loss` (e.g. 1.5). If it starts missing quiet callers, lower it (e.g. 0.7). If val peaks early, retry with `learning_rate: 1e-4`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `read_audio` needs torchaudio or torchcodec | Load audio with soundfile; convert MP3s to WAV first |
| `libcudart.so.13` not found | Reinstall matched torch + torchaudio (2.5.1, cu124) |
| `Missing key aug_prob` (or any key) | Restore `config.yml` from git, edit values only |
| ROC-AUC flat near 0.59 | Set `tune_8k: False` for 16 kHz data |
| `Expected all tensors to be on the same device` | `torch.jit.load(path, map_location="cpu")` |
| ONNX export: module not part of the active trace | Export `m._model` directly; don't wrap it |
| Script finishes instantly | Outputs already existed or every file errored; check counts and errors |
