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

logging.getLogger().setLevel(logging.CRITICAL)

def run_tuning(mode, phase):
    print(f"=== TUNING ALGORITMO: {mode.upper()} (Fase: {phase}) ===")
    data = setup_scenario() # Usa scenario default/storico per il tuning tecnico
    
    base_args = {
        "ball_position": data["ball_position"],
        "player_names": data["starters_home"],
        "phase_name": phase  # <--- AGGIUNTO
    }

    def objective(trial):
        # 1. PARAMETRI ALGORITMO (Questi rimangono globali)
        if "cma" in mode:
            config.CMA_SIGMA_INIT = trial.suggest_float("CMA_SIGMA_INIT", 0.02, 0.2)
            config.CMA_POPSIZE    = trial.suggest_int("CMA_POPSIZE", 10, 40)
        elif mode == "de":
            config.DE_POPSIZE = trial.suggest_int("DE_POPSIZE", 10, 40)
            config.DE_RECOMBINATION = trial.suggest_float("DE_RECOMBINATION", 0.5, 0.9)
            
        # 2. ESECUZIONE
        try:
            if mode == "cma_static":
                best_vec, cost_history = run_cma_static(
                    initial_guess=data["initial_guess"],
                    obstacles=data["obstacles_matrix"],
                    **base_args
                )
                final_fitness = cost_history[-1]
                final_df = flat_to_formation(best_vec, data["starters_home"])
            
            elif mode == "cma_dynamic":
                best_vec, cost_history = run_cma_dynamic(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"],
                    **base_args
                )
                final_fitness = cost_history[-1]
                final_df = flat_to_formation(best_vec, data["starters_home"])

            elif mode == "de":
                best_vec, final_fitness, _ = run_de_optimization(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"],
                    **base_args
                )
                final_df = flat_to_formation(best_vec, data["starters_home"])

            # 3. CONSTRAINT CHECK (Hard)
            # Verifichiamo solo vincoli fisici (sovrapposizioni), non tattici
            check_dict = {"Check": final_df} # Hack per usare penalty_total
            constraint_cost = penalty_total(check_dict)
            
            if constraint_cost > 100.0:
                return 10000.0 # Soluzione non valida fisicamente

            return final_fitness

        except Exception as e:
            return 10000.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30) # Pochi trial per test veloce

    print(f"\n=== MIGLIORI PARAMETRI ALGORITMO ({mode}) ===")
    for k, v in study.best_params.items():
        print(f"{k} = {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["cma_static", "cma_dynamic", "de"])
    parser.add_argument("--phase", type=str, default="Fase difensiva", help="Phase to simulate during tuning")
    
    args = parser.parse_args()
    run_tuning(args.mode, args.phase)