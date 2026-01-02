import sys, torch, os

print("PY:", sys.executable)
print("torch:", torch.__version__, torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
