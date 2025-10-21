import os

def ensure_dirs():
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("data", exist_ok=True)
