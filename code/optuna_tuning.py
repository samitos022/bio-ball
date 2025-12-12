import optuna
import argparse
import numpy as np
import logging
import config
from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from optimization.constraints import penalty_total

from optimization.cma_es import run_optimization as run_cma_static
from optimization.cma_es_dynamic import run_optimization as run_cma_dynamic
from optimization.differential_evolution import run_de_optimization

# Disable verbose logging
logging.getLogger().setLevel(logging.CRITICAL)

def run_tuning(mode, phase):
    print(f"=== ALGORITHM TUNING: {mode.upper()} (Phase: {phase}) ===")
    
    # Use default/historical scenario for technical tuning
    data = setup_scenario() 
    
    base_args = {
        "ball_position": data["ball_position"],
        "player_names": data["starters_home"],
        "phase_name": phase
    }

    def objective(trial):
        # 1. ALGORITHM HYPERPARAMETERS
        if "cma" in mode:
            config.CMA_SIGMA_INIT = trial.suggest_float("CMA_SIGMA_INIT", 0.02, 0.2)
            config.CMA_POPSIZE    = trial.suggest_int("CMA_POPSIZE", 10, 40)
        elif mode == "de":
            config.DE_POPSIZE = trial.suggest_int("DE_POPSIZE", 10, 40)
            config.DE_RECOMBINATION = trial.suggest_float("DE_RECOMBINATION", 0.5, 0.9)
            
        # 2. EXECUTION
        try:
            best_vec = None
            final_fitness = float('inf')

            if mode == "cma_static":
                best_vec, cost_history = run_cma_static(
                    initial_guess=data["initial_guess"],
                    obstacles=data["obstacles_matrix"],
                    **base_args
                )
                final_fitness = cost_history[-1]
            
            elif mode == "cma_dynamic":
                best_vec, cost_history = run_cma_dynamic(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"],
                    **base_args
                )
                final_fitness = cost_history[-1]

            elif mode == "de":
                best_vec, final_fitness, _ = run_de_optimization(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"],
                    **base_args
                )

            # Convert to DataFrame for constraint checking
            final_df = flat_to_formation(best_vec, data["starters_home"])

            # 3. HARD CONSTRAINT CHECK
            # We verify only physical constraints (collisions, boundaries), not tactical ones.
            # 'Check' key is a workaround to use penalty_total on a single frame.
            check_dict = {"Check": final_df} 
            constraint_cost = penalty_total(check_dict)
            
            # If physical constraints are violated significantly, penalize heavily
            if constraint_cost > 100.0:
                return 10000.0 

            return final_fitness

        except Exception as e:
            # Return high penalty on crash
            return 10000.0

    # Run Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print(f"\n=== BEST ALGORITHM PARAMETERS ({mode}) ===")
    for k, v in study.best_params.items():
        print(f"{k} = {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune algorithm hyperparameters (technical tuning).")
    parser.add_argument("--mode", type=str, required=True, choices=["cma_static", "cma_dynamic", "de"])
    parser.add_argument("--phase", type=str, default="Defensive phase", help="Phase to simulate during tuning")
    
    args = parser.parse_args()
    run_tuning(args.mode, args.phase)