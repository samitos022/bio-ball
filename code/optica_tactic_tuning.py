import optuna
import json
import numpy as np
import config
from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from optimization.cma_es import run_optimization as run_cma_static
import logging

logging.getLogger().setLevel(logging.CRITICAL)

# =============================================================================
# CONFIGURAZIONE SCENARIO DA JSON
# =============================================================================
JSON_FILE = "code/data/formations/ground_truth.json"

# Inserisci qui il nome esatto che hai salvato nel Formation Creator
SCENARIO_NAME = "Difesa_Pressata" 

# 1. Caricamento Dati JSON
try:
    with open(JSON_FILE, "r") as f:
        full_db = json.load(f)
    
    if SCENARIO_NAME not in full_db:
        raise ValueError(f"Scenario '{SCENARIO_NAME}' non trovato nel JSON.")
    
    scenario_data = full_db[SCENARIO_NAME]
    print(f"=== CARICAMENTO SCENARIO: {SCENARIO_NAME} ===")

except Exception as e:
    print(f"Errore critico: {e}")
    exit()

# 2. Estrazione dati dal JSON (Target Home, Custom Ball, Custom Obstacles)
# Home (Target)
target_home_dict = scenario_data["home"]
target_list = []
# Assumiamo ci siano 11 giocatori
for i in range(11):
    idx = str(i)
    if idx in target_home_dict:
        target_list.append(target_home_dict[idx])
    else:
        # Fallback se manca un giocatore nel json (non dovrebbe succedere)
        target_list.append([0.5, 0.5]) 
TARGET_POSITIONS = np.array(target_list)

# Ball (Custom)
CUSTOM_BALL_POS = np.array(scenario_data["ball"])

# Away / Obstacles (Custom)
obstacles_dict = scenario_data["away"]
obs_list = []
for i in range(len(obstacles_dict)):
    idx = str(i)
    if idx in obstacles_dict:
        obs_list.append(obstacles_dict[idx])
CUSTOM_OBSTACLES = np.array(obs_list)

print(f"Palla impostata a: {CUSTOM_BALL_POS}")
print(f"Avversari caricati: {len(CUSTOM_OBSTACLES)}")


# 3. Setup Base (Solo per nomi giocatori e struttura iniziale)
data = setup_scenario() # Carica i dati generici per avere i nomi e i riferimenti
INITIAL_GUESS = data["initial_guess"] # Partiamo dalla formazione media standard (o potremmo randomizzare)
PLAYER_NAMES = data["starters_home"]


def objective(trial):
    # =========================================================================
    # OPTUNA MODIFICA LA TATTICA
    # =========================================================================
    
    # Tuning Obiettivi Tattici
    config.OBJ_W_COVER   = trial.suggest_float("OBJ_W_COVER", 0.0, 10.0)
    config.OBJ_W_PASS    = trial.suggest_float("OBJ_W_PASS", 0.0, 10.0)
    config.OBJ_W_BALL    = trial.suggest_float("OBJ_W_BALL", 0.0, 50.0)
    config.OBJ_W_OFFSIDE = trial.suggest_float("OBJ_W_OFFSIDE", 10.0, 200.0)
    
    # Tuning Parametri Interni (Fisica del gioco)
    config.PASS_W_BLOCK  = trial.suggest_float("PASS_W_BLOCK", 1.0, 20.0)
    config.PASS_MAX_LEN  = trial.suggest_float("PASS_MAX_LEN", 0.2, 0.5)

    # =========================================================================
    # ESECUZIONE SIMULAZIONE
    # =========================================================================
    try:
        # IMPORTANTE: Passiamo gli ostacoli e la palla del JSON, non quelli del dataset!
        best_vec, cost_history = run_cma_static(
            initial_guess=INITIAL_GUESS,
            obstacles=CUSTOM_OBSTACLES,  # <--- Usiamo gli avversari disegnati
            ball_position=CUSTOM_BALL_POS, # <--- Usiamo la palla disegnata
            player_names=PLAYER_NAMES
        )
        
        # =====================================================================
        # CALCOLO LOSS (MSE vs TARGET)
        # =====================================================================
        # df_opt = flat_to_formation(best_vec, PLAYER_NAMES)
        opt_positions = best_vec[['x', 'y']].values
        
        # Calcoliamo quanto la formazione generata assomiglia a quella disegnata (rossa)
        mse = np.mean(np.linalg.norm(opt_positions - TARGET_POSITIONS, axis=1)**2)
        
        return mse

    except Exception as e:
        # Penalizza pesantemente se crasha o parametri impossibili
        print(f"\n[ERRORE TRIAL] {e}")
        return 1000.0

if __name__ == "__main__":
    print("Avvio ricerca pesi tattici...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    
    print("\n=== MIGLIORI PESI PER REPLICARE LO SCENARIO ===")
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Errore medio (MSE): {study.best_value:.6f}")
    for k, v in study.best_params.items():
        print(f"{k} = {v}")
    
    # Salva su file
    with open(f"best_tactics_{SCENARIO_NAME}.txt", "w") as f:
        f.write(f"# Tattica per: {SCENARIO_NAME}\n")
        for k, v in study.best_params.items():
            f.write(f"{k} = {v}\n")