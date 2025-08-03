import os
import joblib

def save_checkpoint(obj, name):
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"{name}.pkl")
    joblib.dump(obj, path)

def load_checkpoint(name):
    path = os.path.join("checkpoints", f"{name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {"best_params": [], "total_time": 0}
