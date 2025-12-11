import optuna
import json
import argparse
import numpy as np
import config
from utils.setup import setup_scenario
from optimization.cma_es import run_optimization as run_cma_static
import logging

# Disabilita log verbosi
logging.getLogger().setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser(description="Optuna Tactic Tuning")

parser.add_argument("--scenario", type=str, required=True,
                    help="Name of the scenario in ground_truth.json (Target)")
parser.add_argument("--phase", type=str, default="Fase_difensiva",
                    choices=["Fase_difensiva", "Possesso_offensivo", "Possesso_difensivo"],
                    help="Which phase weights to tune")

args = parser.parse_args()

JSON_FILE = "code/data/formations/ground_truth.json"

# 1. Caricamento Dati JSON e Setup
try:
    # Setup base per nomi e struttura
    data = setup_scenario(scenario_name=args.scenario)
    
    # Verifica che lo scenario esista nel JSON per il target
    with open(JSON_FILE, "r") as f:
        full_db = json.load(f)
    
    if args.scenario not in full_db:
        raise ValueError(f"Scenario '{args.scenario}' non trovato.")
    
    scenario_data = full_db[args.scenario]
    
    # Target Home (La formazione disegnata da copiare)
    target_home_dict = scenario_data["home"]
    target_list = []
    # Assumiamo ordine 0-10
    for i in range(len(data["starters_home"])):
        idx = str(i)
        if idx in target_home_dict:
            target_list.append(target_home_dict[idx])
        else:
            # Fallback se manca un giocatore nel disegno
            target_list.append(data["df_home_start"].iloc[i][['x', 'y']].values)
            
    TARGET_POSITIONS = np.array(target_list)
    phase = args.phase.replace('_', ' ')
    print(f"=== TUNING TATTICO PER FASE: {phase} ===")
    print(f"Target Scenario: {args.scenario}")

except Exception as e:
    print(f"Errore critico setup: {e}")
    exit()


def objective(trial):
    # Accediamo al dizionario della fase specifica
    phase_weights = config.PHASE_WEIGHTS[phase]

    # --- TUNING DINAMICO IN BASE ALLA FASE ---
    
    if phase == "Fase difensiva":
        # Tuniamo i pesi difensivi
        phase_weights["W_COVERAGE"]    = trial.suggest_float("W_COVERAGE", 0.1, 3.0)
        phase_weights["W_BALL_PRESS"]  = trial.suggest_float("W_BALL_PRESS", 0.1, 100.0)
        phase_weights["W_MARKING"]    = trial.suggest_float("W_MARKING", 0.1, 100.0)
        phase_weights["W_COMPACTNESS"]     = trial.suggest_float("W_COMPACTNESS", 0.1, 100.0) # Priorità alta
        phase_weights["W_LINE_HEIGHT"]  = trial.suggest_float("W_LINE_HEIGHT", 0.1, 100.0)
        phase_weights["W_PREV_MARKING"]  = trial.suggest_float("W_PREV_MARKING", 0.1, 3.0)
        
        
    elif phase == "Possesso offensivo":
        # Tuniamo i pesi offensivi
        phase_weights["W_MARKING"]     = trial.suggest_float("W_MARKING", 0.0, 10.0)
        phase_weights["W_COVERAGE"]    = trial.suggest_float("W_COVERAGE", 5.0, 50.0)
        phase_weights["W_PASSING"]     = trial.suggest_float("W_PASSING", 1.0, 20.0)
        phase_weights["W_OFFSIDE"]     = trial.suggest_float("W_OFFSIDE", 10.0, 100.0)
        phase_weights["W_BALL_PRESS"]  = trial.suggest_float("W_BALL_PRESS", 0.0, 10.0) # Ball support basso

    elif phase == "Possesso difensivo":
        phase_weights["W_COVERAGE"]    = trial.suggest_float("W_COVERAGE", 0.1, 100.0)
        phase_weights["W_PASSING"]     = trial.suggest_float("W_PASSING", 0.1, 100.0) # Priorità alta
        phase_weights["W_BALL_PRESS"]  = trial.suggest_float("W_BALL_PRESS", 0.1, 100.0)
        phase_weights["W_MARKING"]    = trial.suggest_float("W_MARKING", 0.1, 100.0)
        phase_weights["W_COMPACTNESS"]     = trial.suggest_float("W_COMPACTNESS", 0.1, 100.0) # Priorità alta
        phase_weights["W_LINE_HEIGHT"]  = trial.suggest_float("W_LINE_HEIGHT", 0.1, 100.0)
        phase_weights["W_PREV_MARKING"]  = trial.suggest_float("W_PREV_MARKING", 0.1, 100.0)

    # Tuning parametri fisici (comuni, opzionale)
    config.PASS_W_BLOCK = trial.suggest_float("PASS_W_BLOCK", 1.0, 15.0)

    try:
        # Eseguiamo l'ottimizzazione con i nuovi pesi
        best_vec, _ = run_cma_static(
            initial_guess=data["initial_guess"],
            obstacles=data["obstacles_matrix"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=phase  # <--- Importante: passiamo la fase corretta
        )
        
        # CALCOLO LOSS (MSE vs TARGET DISEGNATO)
        # Quanto la formazione generata assomiglia a quella disegnata?
        
        # best_vec è un array numpy flat o reshape, assicuriamoci sia matrice
        # run_cma_static ritorna un vettore flat (in cma_es.py originale)
        opt_positions = best_vec.reshape(-1, 2)
        
        # MSE
        mse = np.mean(np.linalg.norm(opt_positions - TARGET_POSITIONS, axis=1)**2)
        
        return mse

    except Exception as e:
        return 1000.0 # Penalità per crash

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    print("Avvio ricerca pesi (20 trials)...")
    study.optimize(objective, n_trials=100)
    
    print(f"\n=== MIGLIORI PESI PER {phase.upper()} ===")
    print(f"Best MSE: {study.best_value:.6f}")
    
    filename = f"best_params_{args.scenario}.txt"
    with open(filename, "w") as f:
        f.write(f"# Parametri ottimizzati per scenario: {args.scenario}\n")
        f.write(f"# Fase: {phase}\n\n")
        f.write(f'    "{phase}": \n')
        for k, v in study.best_params.items():
            f.write(f'        "{k}": {v:.4f},\n')
        f.write("    },\n")
    
    print(f"Salvati in {filename} (Copia incolla in config.py)")