import optuna
import json
import argparse
import numpy as np
import logging
import config
from utils.setup import setup_scenario
from optimization.cma_es import run_optimization as run_cma_static

# Disable verbose logging
logging.getLogger().setLevel(logging.CRITICAL)

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Optuna Tactic Tuning (Reverse Engineering)")

parser.add_argument("--scenario", type=str, required=True,
                    help="Name of the scenario in ground_truth.json (Target formation)")
parser.add_argument("--phase", type=str, default="Defensive_phase",
                    choices=["Defensive_phase", "Attacking_possession", "Defensive_possession"],
                    help="Which game phase weights to tune")

args = parser.parse_args()

# Constants
JSON_FILE = "code/data/formations/ground_truth.json"
PHASE_NAME = args.phase.replace('_', ' ')  # Convert CLI format to Config format

# --- DATA SETUP ---
try:
    # 1. Load base scenario data (Player names, etc.)
    data = setup_scenario(scenario_name=args.scenario)
    
    # 2. Load Ground Truth JSON
    with open(JSON_FILE, "r") as f:
        full_db = json.load(f)
    
    if args.scenario not in full_db:
        raise ValueError(f"Scenario '{args.scenario}' not found in JSON.")
    
    scenario_data = full_db[args.scenario]
    
    # 3. Extract Target Home Positions (The "drawing" we want to mimic)
    target_home_dict = scenario_data["home"]
    target_list = []
    
    # Ensure correct order (0-10) matching the starters list
    for i in range(len(data["starters_home"])):
        idx = str(i)
        if idx in target_home_dict:
            target_list.append(target_home_dict[idx])
        else:
            # Fallback: use historical average position if player is missing in drawing
            target_list.append(data["df_home_start"].iloc[i][['x', 'y']].values)
            
    TARGET_POSITIONS = np.array(target_list)
    
    print(f"=== TACTIC TUNING FOR PHASE: {PHASE_NAME} ===")
    print(f"Target Scenario: {args.scenario}")

except Exception as e:
    print(f"Critical Setup Error: {e}")
    exit()


def objective(trial):
    """
    Optuna objective function.
    Adjusts weights to minimize the distance (MSE) between the generated formation
    and the target ground truth formation.
    """
    # Access the specific dictionary for the current phase
    phase_weights = config.PHASE_WEIGHTS[PHASE_NAME]

    # --- DYNAMIC WEIGHT SUGGESTION ---
    
    if PHASE_NAME == "Defensive phase":
        phase_weights["W_COVERAGE"]     = trial.suggest_float("W_COVERAGE", 0.1, 3.0)
        phase_weights["W_BALL_PRESS"]   = trial.suggest_float("W_BALL_PRESS", 0.1, 100.0)
        phase_weights["W_MARKING"]      = trial.suggest_float("W_MARKING", 0.1, 100.0)
        phase_weights["W_COMPACTNESS"]  = trial.suggest_float("W_COMPACTNESS", 0.1, 100.0)
        phase_weights["W_LINE_HEIGHT"]  = trial.suggest_float("W_LINE_HEIGHT", 0.1, 100.0)
        phase_weights["W_PREV_MARKING"] = trial.suggest_float("W_PREV_MARKING", 0.1, 3.0)
        
    elif PHASE_NAME == "Attacking possession":
        phase_weights["W_MARKING"]      = trial.suggest_float("W_MARKING", 0.0, 10.0)
        phase_weights["W_COVERAGE"]     = trial.suggest_float("W_COVERAGE", 5.0, 50.0)
        phase_weights["W_PASSING"]      = trial.suggest_float("W_PASSING", 1.0, 20.0)
        phase_weights["W_OFFSIDE"]      = trial.suggest_float("W_OFFSIDE", 10.0, 100.0)
        phase_weights["W_BALL_PRESS"]   = trial.suggest_float("W_BALL_PRESS", 0.0, 10.0)

    elif PHASE_NAME == "Defensive possession":
        phase_weights["W_COVERAGE"]     = trial.suggest_float("W_COVERAGE", 0.1, 100.0)
        phase_weights["W_PASSING"]      = trial.suggest_float("W_PASSING", 0.1, 100.0)
        phase_weights["W_BALL_PRESS"]   = trial.suggest_float("W_BALL_PRESS", 0.1, 100.0)
        phase_weights["W_MARKING"]      = trial.suggest_float("W_MARKING", 0.1, 100.0)
        phase_weights["W_COMPACTNESS"]  = trial.suggest_float("W_COMPACTNESS", 0.1, 100.0)
        phase_weights["W_LINE_HEIGHT"]  = trial.suggest_float("W_LINE_HEIGHT", 0.1, 100.0)
        phase_weights["W_PREV_MARKING"] = trial.suggest_float("W_PREV_MARKING", 0.1, 100.0)

    # Tune common physical parameters
    config.PASS_W_BLOCK = trial.suggest_float("PASS_W_BLOCK", 1.0, 15.0)

    try:
        # Run Optimization with suggested weights
        best_vec, _ = run_cma_static(
            initial_guess=data["initial_guess"],
            obstacles=data["obstacles_matrix"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=PHASE_NAME
        )
        
        # Calculate Loss (MSE vs Target)
        # Reshape flat vector to (N, 2)
        opt_positions = best_vec.reshape(-1, 2)
        
        # Mean Squared Error between generated positions and drawing
        mse = np.mean(np.linalg.norm(opt_positions - TARGET_POSITIONS, axis=1)**2)
        
        return mse

    except Exception as e:
        # Return high penalty if simulation crashes
        return 1000.0

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    print("Starting parameter search (20 trials)...")
    study.optimize(objective, n_trials=20)
    
    print(f"\n=== BEST PARAMETERS FOR {PHASE_NAME.upper()} ===")
    print(f"Best MSE: {study.best_value:.6f}")
    
    filename = f"best_params_{args.scenario}.txt"
    with open(filename, "w") as f:
        f.write(f"# Optimized Parameters for Scenario: {args.scenario}\n")
        f.write(f"# Phase: {PHASE_NAME}\n\n")
        f.write(f'    "{PHASE_NAME}": \n')
        for k, v in study.best_params.items():
            f.write(f'        "{k}": {v:.4f},\n')
        f.write("    \n")
    
    print(f"Parameters saved to {filename} (Copy to config.py)")