import torch
m = torch.jit.load("/home/justdial/change_2/voice_r/silero_vad_tuned_8k.jit", map_location="cpu")
print("=== top-level forward ===");  print(m.code)
print("=== _model (16k) forward ==="); print(m._model.code)
for name, sub in m._model.named_children():
    print("child:", name, type(sub).__name__)