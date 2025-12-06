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

# Disabilita log troppo verbosi, vogliamo vedere solo Optuna
logging.getLogger().setLevel(logging.CRITICAL)

def run_tuning(mode):
    print(f"=== CARICAMENTO DATI PER {mode.upper()} ===")
    data = setup_scenario()
    
    # Argomenti base comuni
    base_args = {
        "ball_position": data["ball_position"],
        "player_names": data["starters_home"]
    }

    def objective(trial):
        # 1. PARAMETRI COMUNI (FISICA E REGOLE INTERNE)
        # Questi parametri influenzano LA FUNZIONE DI COSTO, quindi valgono per tutti.
        # Cerchiamo il bilanciamento per rendere la superficie di ricerca "liscia".
        
        config.PASS_BLOCK_THRESHOLD = trial.suggest_float("PASS_BLOCK_THRESHOLD", 0.02, 0.06)
        config.PASS_W_BLOCK         = trial.suggest_float("PASS_W_BLOCK", 5.0, 15.0)
        config.PENALTY_W_PROXIMITY  = trial.suggest_float("PENALTY_W_PROXIMITY", 100.0, 800.0)
        
        # Opzionale: soglia passaggio lungo
        config.PASS_MAX_LEN = trial.suggest_float("PASS_MAX_LEN", 0.30, 0.45)

        # 2. PARAMETRI SPECIFICI DEL SOLVER (IL "CERVELLO")
        
        # --- A. CMA-ES (Statico o Dinamico) ---
        if "cma" in mode:
            config.CMA_SIGMA_INIT = trial.suggest_float("CMA_SIGMA_INIT", 0.02, 0.2)
            config.CMA_POPSIZE    = trial.suggest_int("CMA_POPSIZE", 12, 32)
            # Tolleranza: se troppo bassa ci mette troppo, se troppo alta è impreciso
            config.CMA_TOLFUN     = trial.suggest_float("CMA_TOLFUN", 1e-4, 1e-2, log=True)
            
        # --- B. DIFFERENTIAL EVOLUTION ---
        elif mode == "de":
            config.DE_POPSIZE = trial.suggest_int("DE_POPSIZE", 10, 30)
            config.DE_RECOMBINATION = trial.suggest_float("DE_RECOMBINATION", 0.5, 0.9)
            # Mutation in DE è una tupla (min, max). Optuna suggerisce un valore centrale.
            mut_center = trial.suggest_float("DE_MUTATION_CENTER", 0.4, 0.9)
            mut_width = 0.2
            config.DE_MUTATION = (max(0, mut_center - mut_width), min(1.0, mut_center + mut_width))

        # 3. ESECUZIONE ALGORITMO
        try:
            best_vec = None
            final_fitness = float('inf')
            final_df = None

            if mode == "cma_es":
                final_df, cost_history = run_cma_static(
                    initial_guess=data["initial_guess"],
                    obstacles=data["obstacles_matrix"],  # Static obstacles
                    **base_args
                )
                final_fitness = cost_history[-1]
                # CMA ritorna già il DF, non il vettore
            
            elif mode == "cma_es_dynamic":
                final_df, cost_history = run_cma_dynamic(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"], # Dynamic base
                    **base_args
                )
                final_fitness = cost_history[-1]

            elif mode == "de":
                # DE ritorna il vettore raw
                best_vec, final_fitness, _ = run_de_optimization(
                    initial_guess=data["initial_guess"],
                    initial_away_df=data["df_away_start"],
                    **base_args
                )
                final_df = flat_to_formation(best_vec, data["starters_home"])

            # =================================================================
            # 4. CONTROLLO SICUREZZA (HARD CONSTRAINTS CHECK)
            # =================================================================
            # Optuna potrebbe barare abbassando troppo PENALTY_W_PROXIMITY.
            # Se la soluzione finale ha giocatori sovrapposti, penalizziamo il trial.
            
            # Usiamo penalty_total passando solo la formazione finale come "Start"
            # (trucco per verificare collisioni interne e boundaries di un singolo frame)
            # Nota: penalty_total si aspetta un dict di frames.
            constraint_check_cost = penalty_total({"Check": final_df})
            
            # Se viola i vincoli fisici (es. > 100), diamo un costo enorme
            if constraint_check_cost > 200.0:
                return 10000.0 + constraint_check_cost

            return final_fitness

        except Exception as e:
            # Se l'algoritmo crasha con parametri assurdi
            print(f"[TRIAL FAILED] {e}")
            return 10000.0

    # Creazione Studio
    study = optuna.create_study(direction="minimize")
    print(f"Avvio Tuning Optuna per: {mode.upper()}...")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n=== RISULTATI MIGLIORI ({mode}) ===")
    print(f"Best Fitness: {study.best_value:.4f}")
    
    filename = f"best_params_{mode}.txt"
    with open(filename, "w") as f:
        f.write(f"# Parametri ottimizzati per {mode}\n")
        for key, value in study.best_params.items():
            print(f"{key} = {value}")
            # Se è la mutazione (tupla) dobbiamo ricostruirla per scriverla bene
            if key == "DE_MUTATION_CENTER":
                 mut_width = 0.2
                 val_min = max(0, value - mut_width)
                 val_max = min(1.0, value + mut_width)
                 f.write(f"DE_MUTATION = ({val_min:.2f}, {val_max:.2f})\n")
            else:
                f.write(f"{key} = {value}\n")
    
    print(f"Salvato in {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning for Bio-Ball")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["cma_es", "cma_es_dynamic", "de"],
                        help="Choose which algorithm to tune")
    
    args = parser.parse_args()
    run_tuning(args.mode)