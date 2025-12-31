import os
import glob
import pandas as pd
import numpy as np
import time
import json

# --- MLX SETUP ---
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    print("✅ Wykryto Apple Silicon (MLX). Jedziemy z koksem.")
except ImportError:
    print("❌ BŁĄD: Brak biblioteki MLX. Zainstaluj: pip install mlx")
    exit()

# KONFIGURACJA
DATA_FOLDER = "treningi_csv"  # Tutaj wrzuć pliki CSV lub JSON
MODEL_FILE = "cycling_brain_weights.npz"
HISTORY_FILE = "brain_evolution_history.json"

# --- DEFINICJA MODELU (Musi być identyczna jak w app.py) ---
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

# --- FUNKCJE POMOCNICZE ---

def load_data(filepath):
    """Smart Loader: Radzi sobie z zagnieżdżonymi JSONami i CSV"""
    file_ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    
    try:
        if file_ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # PRZYPADEK 1: JSON to po prostu lista rekordów (Idealnie)
            if isinstance(data, list):
                df = pd.DataFrame(data)
                
            # PRZYPADEK 2: JSON to słownik (Dict)
            elif isinstance(data, dict):
                # Szukamy klucza, który trzyma "mięso" (najdłuższą listę)
                # Częste nazwy w apkach sportowych:
                candidates = ['samples', 'data', 'records', 'trackPoints', 'points', 'streams', 'rows']
                
                target_list = None
                
                # A. Szukamy po znanych nazwach
                for key in candidates:
                    if key in data and isinstance(data[key], list):
                        target_list = data[key]
                        break
                
                # B. Jeśli nie znaleziono po nazwie, szukamy NAJDŁUŻSZEJ listy w całym JSON-ie
                if target_list is None:
                    max_len = 0
                    for k, v in data.items():
                        if isinstance(v, list) and len(v) > max_len:
                            # Dodatkowe zabezpieczenie: lista musi zawierać słowniki (rekordy)
                            if len(v) > 0 and isinstance(v[0], dict):
                                target_list = v
                                max_len = len(v)
                
                # C. Jeśli nadal nic, może to format kolumnowy? {'time': [1,2], 'watts': [100, 200]}
                # Próbujemy stworzyć DF bezpośrednio, ale bezpiecznie
                if target_list is None:
                    try:
                        # dict_of_lists
                        df = pd.DataFrame.from_dict(data, orient='columns')
                        # Jeśli zadziałało, ale kolumny mają różne długości, Pandas rzuci błąd, który złapiemy niżej
                    except ValueError:
                        # Ostatnia deska ratunku: json_normalize na płasko
                        df = pd.json_normalize(data)
                else:
                    # Mamy naszą listę!
                    df = pd.json_normalize(target_list)

        else:
            # Ładowanie CSV / TXT (Stara metoda)
            try:
                df = pd.read_csv(filepath, low_memory=False)
            except:
                df = pd.read_csv(filepath, sep=';', low_memory=False)
    
        # --- CZYSZCZENIE FINALNE ---
        if 'df' in locals() and not df.empty:
            # Normalizacja nazw kolumn (małe litery, bez spacji)
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Fix dla TymeWear / json nested (czasami dane są w 'data.watts', 'data.time')
            # Usuwamy prefixy typu 'data.' z nazw kolumn
            df.columns = [c.split('.')[-1] for c in df.columns]
            
            return df
        else:
            print(f"   -> ⚠️ Pusty lub nieczytelny plik: {filename}")
            return pd.DataFrame()

    except Exception as e:
        print(f"   -> ⚠️ Krytyczny błąd odczytu {filename}: {e}")
        return pd.DataFrame()

def process_data(df):
    if df.empty: return df

    # Minimalna obróbka potrzebna do treningu
    if 'time' not in df.columns:
        # Jeśli brak czasu, tworzymy sztuczny
        df['time'] = np.arange(len(df)).astype(float)
    
    # Sortowanie i czyszczenie
    df = df.sort_values('time').reset_index(drop=True)
    
    # Konwersja kolumn numerycznych (dla bezpieczeństwa, zwłaszcza przy JSON)
    cols_to_numeric = ['watts', 'heartrate', 'cadence', 'time']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Uzupełnianie dziur (Interpolacja)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()

    # Smoothing (kluczowe dla sieci neuronowej)
    window = 30 # 30 sekund
    if 'watts' in df.columns:
        df['watts_smooth'] = df['watts'].rolling(window=window, min_periods=1).mean()
    if 'heartrate' in df.columns:
        df['heartrate_smooth'] = df['heartrate'].rolling(window=window, min_periods=1).mean()
    if 'cadence' in df.columns:
        df['cadence_smooth'] = df['cadence'].rolling(window=window, min_periods=1).mean()
    
    df['time_min'] = df['time'] / 60.0
    return df

def filter_and_prepare(df, target_watts, tolerance=15, min_samples=30):
    """
    Filtruje dane tylko dla konkretnego zakresu mocy.
    Np. dla 280W bierze zakres 265W-295W.
    """
    if df.empty or 'watts_smooth' not in df.columns:
        return None, None

    # Maska: szukamy momentów, gdzie moc była blisko celu
    mask = (df['watts_smooth'] >= target_watts - tolerance) & \
           (df['watts_smooth'] <= target_watts + tolerance)
    
    # Jeśli mamy za mało danych (np. mniej niż 30 sekund w strefie), odpuszczamy
    if mask.sum() < min_samples:
        return None, None

    df_filtered = df[mask].copy()

    # Przygotowanie Tensorów MLX
    w = df_filtered['watts_smooth'].values / 500.0
    c = df_filtered['cadence_smooth'].values / 120.0 if 'cadence_smooth' in df_filtered else np.zeros_like(w)
    t = df_filtered['time_min'].values / df['time_min'].max() # Normalizacja czasem całego treningu
    y_target = df_filtered['heartrate_smooth'].values / 200.0

    X_np = np.column_stack((w, c, t)).astype(np.float32)
    y_np = y_target.astype(np.float32).reshape(-1, 1)

    return mx.array(X_np), mx.array(y_np)

def update_history(hr_base, hr_thresh, filename):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except: pass
    
    entry = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "source_file": filename,
        # Zapisujemy None, jeśli brak danych (JSON to przyjmie jako null)
        "hr_base": float(hr_base) if hr_base is not None else None,
        "hr_thresh": float(hr_thresh) if hr_thresh is not None else None
    }
    history.append(entry)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def train_loop():
    # 1. Szukamy plików
    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    files += glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
    files += glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not files:
        print(f"⚠️ Nie znaleziono plików w folderze '{DATA_FOLDER}'.")
        return

    print(f"📂 Znaleziono {len(files)} treningów. Sortuję chronologicznie...")
    files.sort() 

    # Inicjalizacja modelu (Globalnego)
    model = PhysioNet()
    mx.eval(model.parameters())
    
    # Funkcja straty i optimizer
    def mse_loss(pred, target): return mx.mean((pred - target) ** 2)
    optimizer = optim.Adam(learning_rate=0.02) # Nieco agresywniejsze uczenie dla wycinków
    
    def train_step(model, X, y):
        pred = model(X)
        loss = mse_loss(pred, y)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, train_step)
    
    targets = {
        "BASE": 280,   # Cel 1: Baza
        "THRESH": 360  # Cel 2: Próg
    }

    total_start = time.time()
    
    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"\n[{idx+1}/{len(files)}] Analiza: {filename}")
        
        try:
            df_raw = load_data(file_path)
            if df_raw.empty: continue

            df = process_data(df_raw)
            if len(df) < 100: continue

            results = {} # Tu zbierzemy wyniki dla tego pliku

            # --- NOWA LOGIKA: Trenujemy osobno dla każdego celu ---
            for name, watts in targets.items():
                
                # 1. Filtrujemy dane (tylko momenty, gdzie jechałeś ~Watts)
                X_chunk, y_chunk = filter_and_prepare(df, watts, tolerance=15, min_samples=60)
                
                if X_chunk is not None:
                    # Resetujemy wagi modelu do stanu globalnego (lub trenujemy dalej - tu decydujemy się na fine-tuning)
                    # W tym skrypcie robimy fine-tuning ciągły, ale na przefiltrowanych danych model "przypomni sobie" konkretną strefę
                    
                    # Szybki trening na tym wycinku (Overfitting jest tu pożądany, bo chcemy odwzorować TEN trening)
                    for _ in range(100): 
                        loss, grads = loss_and_grad_fn(model, X_chunk, y_chunk)
                        optimizer.update(model, grads)
                        mx.eval(model.parameters(), optimizer.state)
                    
                    # Predykcja
                    # Parametry wejściowe: [Moc, Kadencja, Czas(połowa treningu)]
                    # Kadencję przyjmujemy optymalną (85-90) lub średnią z wycinka
                    cadence_norm = 90.0/120.0 
                    in_tensor = mx.array([[watts/500.0, cadence_norm, 0.5]])
                    
                    pred_hr = float(model(in_tensor)[0][0]) * 200.0
                    results[name] = pred_hr
                    print(f"   -> {name} ({watts}W): {pred_hr:.1f} bpm (znaleziono dane)")
                
                else:
                    results[name] = None
                    print(f"   -> {name} ({watts}W): Brak danych w tym treningu.")

            # Zapisujemy wynik do historii
            update_history(results["BASE"], results["THRESH"], filename)

        except Exception as e:
            print(f"   -> 💥 Błąd: {e}")

    print("-" * 30)
    print("Zapisano historię ewolucji formy.")

    # Spłaszczanie i zapis
    params = model.parameters()
    flat_params = {}
    for layer_name, layer_params in params.items():
        if isinstance(layer_params, dict):
            for param_name, param_value in layer_params.items():
                flat_params[f"{layer_name}.{param_name}"] = param_value
        else:
            flat_params[layer_name] = layer_params
    mx.savez(MODEL_FILE, **flat_params)
    
    total_time = time.time() - total_start
    print(f"🚀 GOTOWE! Przemielono {len(files)} plików w {total_time:.1f} sekund.")

if __name__ == "__main__":
    train_loop()