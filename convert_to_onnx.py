"""
Export the fine-tuned 16k head (_model) of a Silero VAD JIT model to ONNX,
then verify ONNX matches JIT chunk-by-chunk on real audio.

ONNX interface produced:
  inputs : input [batch, samples] float32, h [2, batch, H] float32, c [2, batch, H] float32
  outputs: output [batch, 1], hn, cn   (feed hn/cn back as h/c on the next chunk)
"""
import glob, numpy as np, soundfile as sf, torch, onnxruntime as ort

BASE = "/home/justdial/change_2/voice_r"
JIT = f"{BASE}/silero_vad_tuned_8k.jit"          # contains the tuned 16k head (tune_8k: False run)
ONNX = f"{BASE}/silero_vad_tuned_16k.onnx"
OLD_ONNX = f"{BASE}/silero_vad.onnx"
SR, CHUNK = 16000, 512
TEST_WAV = sorted(glob.glob(f"{BASE}/dataset_dir/mixed/val/*.wav"))[0]

m = torch.jit.load(JIT, map_location="cpu").eval()
sub = m._model                                    # 16k head

# 1. read the real h/c shapes from the model instead of guessing
with torch.no_grad():
    _, h_tmp, c_tmp = sub(torch.zeros(1, CHUNK))   # uses the model's default zero state
h0, c0 = torch.zeros_like(h_tmp), torch.zeros_like(c_tmp)

# 2. export the scripted submodule directly (no tracing wrapper needed)
x = torch.zeros(1, CHUNK)
torch.onnx.export(
    sub, (x, h0, c0), ONNX,
    input_names=["input", "h", "c"], output_names=["output", "hn", "cn"],
    dynamic_axes={"input": {0: "batch", 1: "samples"},
                  "h": {1: "batch"}, "c": {1: "batch"},
                  "output": {0: "batch"}, "hn": {1: "batch"}, "cn": {1: "batch"}},
    opset_version=16, do_constant_folding=True,
)
print("exported:", ONNX)

# 3. verify: same audio, chunk by chunk, state carried forward in both
wav, sr = sf.read(TEST_WAV, dtype="float32")
assert sr == SR, f"test file is {sr} Hz"
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
m.reset_states()
h, c = h0.numpy(), c0.numpy()
jit_p, onnx_p = [], []
with torch.no_grad():
    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        chunk = wav[i:i + CHUNK][None, :]
        jit_p.append(m(torch.from_numpy(chunk), SR).item())
        out, h, c = sess.run(None, {"input": chunk, "h": h, "c": c})
        onnx_p.append(float(out[0, 0]))
diff = np.abs(np.array(jit_p) - np.array(onnx_p)).max()
print(f"verified on {len(jit_p)} chunks of {TEST_WAV.split('/')[-1]} | max |jit - onnx| = {diff:.2e}",
      "-> OK" if diff < 1e-4 else "-> MISMATCH, do not deploy")

# 4. compare with the interface of the ONNX currently used in production
for name, path in [("current prod", OLD_ONNX), ("new tuned", ONNX)]:
    try:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"\n{name}: {path}")
        print("  inputs :", [(i.name, i.shape) for i in s.get_inputs()])
        print("  outputs:", [(o.name, o.shape) for o in s.get_outputs()])
    except Exception as e:
        print(f"\n{name}: could not load ({e})")