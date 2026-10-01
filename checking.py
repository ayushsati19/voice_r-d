import torch
a = torch.jit.load("/home/justdial/change_2/voice_r/silero_vad.jit", map_location="cpu")
b = torch.jit.load("/home/justdial/change_2/voice_r/silero_vad_tuned_8k.jit", map_location="cpu")
sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
changed = [(k, (sa[k] - sb[k]).abs().max().item()) for k in sa if k in sb]
moved = [c for c in changed if c[1] > 1e-6]
print(f"{len(moved)}/{len(changed)} tensors changed")
for k, d in moved[:10]: print(f"  {k}: max diff {d:.2e}")
print("heads:", sorted({k.split('.')[0] for k in sa}))