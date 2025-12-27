import os
import json
import time
import numpy as np
import streamlit as st

MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    MLX_AVAILABLE = True
except ImportError:
    pass

MODEL_FILE = "cycling_brain_weights.npz"
HISTORY_FILE = "brain_evolution_history.json"

if MLX_AVAILABLE:
    class PhysioNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, 1)

        def __call__(self, x):
            x = nn.relu(self.l1(x))
            x = nn.relu(self.l2(x))
            return self.l3(x)

    def save_model(model, filepath):
        flattened_params = {}
        for k, v in model.parameters().items():
            if isinstance(v, dict):
                 for sub_k, sub_v in v.items():
                     flattened_params[f"{k}.{sub_v}"] = sub_v
            else:
                flattened_params[k] = v
        mx.savez(filepath, **dict(mlx.utils.tree_flatten(model.parameters())))

    def load_model(model, filepath):
        if os.path.exists(filepath):
            try:
                weights = mx.load(filepath)
                
                try:
                    model.update(weights)
                except Exception:
                    current_params = model.parameters()
                    new_params = {}
                    
                    for k, v in weights.items():
                        parts = k.split('.')
                        if len(parts) == 2:
                            layer, param = parts
                            if layer not in new_params: new_params[layer] = {}
                            new_params[layer][param] = v
                    
                    model.update(new_params)
                
                return True
            except Exception as e:
                st.sidebar.error(f"⚠️ Błąd AI: {e}")
                print(f"DEBUG ERROR: {e}")
                return False
        return False

    def update_history(hr_base, hr_thresh):
        """Zapisuje historię Baza/Próg do JSON z obsługą None"""
        history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except: pass
        
        entry = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "hr_base": float(hr_base) if hr_base is not None else None,
            "hr_thresh": float(hr_thresh) if hr_thresh is not None else None
        }
        history.append(entry)
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
        return history

    def predict_only(df):
        """Tylko predykcja (bez treningu) - dla automatycznego wykresu"""
        if not os.path.exists(MODEL_FILE):
            return None
            
        w = df['watts_smooth'].values / 500.0
        c = df['cadence_smooth'].values / 120.0 if 'cadence_smooth' in df else np.zeros_like(w)
        t = df['time_min'].values / df['time_min'].max()
        
        X_np = np.column_stack((w, c, t)).astype(np.float32)
        X_np = np.nan_to_num(X_np, copy=False) 
        
        X = mx.array(X_np)
        
        model = PhysioNet()
        if load_model(model, MODEL_FILE):
            y_pred_scaled = model(X)
            return np.array(y_pred_scaled).flatten() * 200.0
        return None
    
    def filter_and_prepare(df, target_watts, tolerance=15, min_samples=30):
        mask = (df['watts_smooth'] >= target_watts - tolerance) & \
            (df['watts_smooth'] <= target_watts + tolerance)
        
        if mask.sum() < min_samples:
            return None, None

        df_filtered = df[mask].copy()
        w = df_filtered['watts_smooth'].values / 500.0
        c = df_filtered['cadence_smooth'].values / 120.0 if 'cadence_smooth' in df_filtered else np.zeros_like(w)
        t = df_filtered['time_min'].values / df['time_min'].max()
        y = df_filtered['heartrate_smooth'].values / 200.0

        X_np = np.column_stack((w, c, t)).astype(np.float32)
        X_np = np.nan_to_num(X_np, copy=False)
        
        y_np = y.astype(np.float32).reshape(-1, 1)
        y_np = np.nan_to_num(y_np, copy=False)

        X = mx.array(X_np)
        Y = mx.array(y_np)
        return X, Y

    def train_cycling_brain(df, epochs=200):
        model = PhysioNet()
        mx.eval(model.parameters())
        
        loaded = load_model(model, MODEL_FILE)
        
        def mse_loss(pred, target): return mx.mean((pred - target) ** 2)
        optimizer = optim.Adam(learning_rate=0.02)
        def train_step(model, X, y):
            loss = mse_loss(model(X), y)
            return loss
        loss_and_grad_fn = nn.value_and_grad(model, train_step)

        status_container = st.empty()
        bar = st.progress(0)
        
        results = {"base": None, "thresh": None}
        targets = [("base", 280), ("thresh", 360)]
        
        status_container.info("Trenowanie modelu ogólnego (cały plik)...")
        w_all = df['watts_smooth'].values / 500.0
        c_all = df['cadence_smooth'].values / 120.0 if 'cadence_smooth' in df else np.zeros_like(w_all)
        t_all = df['time_min'].values / df['time_min'].max()
        y_all = df['heartrate_smooth'].values / 200.0
        
        X_all_np = np.column_stack((w_all, c_all, t_all)).astype(np.float32)
        X_all_np = np.nan_to_num(X_all_np, copy=False)
        
        Y_all_np = y_all.astype(np.float32).reshape(-1, 1)
        Y_all_np = np.nan_to_num(Y_all_np, copy=False)

        X_all = mx.array(X_all_np)
        Y_all = mx.array(Y_all_np)
        
        for i in range(100): 
            loss, grads = loss_and_grad_fn(model, X_all, Y_all)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        
        y_pred_full = np.array(model(X_all)).flatten() * 200.0
        save_model(model, MODEL_FILE) 
        
        # Reset progress bar for next phase
        bar.progress(0)

        step = 0
        total_steps = len(targets) * epochs
        
        for name, watts in targets:
            status_container.info(f"Kalibracja strefy: {watts}W...")
            
            X_chunk, y_chunk = filter_and_prepare(df, watts)
            
            if X_chunk is not None:
                for i in range(epochs):
                    loss, grads = loss_and_grad_fn(model, X_chunk, y_chunk)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    if i % 10 == 0: 
                        step += 10
                        bar.progress(min(step / total_steps, 1.0))
                
                in_vec = mx.array([[watts/500.0, 80.0/120.0, 0.5]]) 
                pred = float(model(in_vec)[0][0]) * 200.0
                results[name] = pred
            else:
                results[name] = None
                step += epochs
                
        bar.empty(); status_container.empty()

        history = update_history(results["base"], results["thresh"])

        return y_pred_full, results["base"], results["thresh"], loaded, history

else:
    def train_cycling_brain(*args, **kwargs): return None, None, None, None, None
    def predict_only(*args, **kwargs): return None
    def update_history(*args, **kwargs): return []
