import os
import joblib

def save_checkpoint(name, object):
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"{name}.pkl")
    joblib.dump(object, path)

def load_checkpoint(name):
    path = os.path.join("checkpoints", f"{name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {"best_params": [], "total_time": 0}
