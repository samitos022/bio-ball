import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Utils
from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from utils.analysis_dynamic import plot_convergence, plot_formation_vertical, plot_formation_with_ball_and_obstacles
from utils.animation import create_evolution_gif
from utils.reporting import print_fitness_breakdown
from utils.away_reaction import react_away_to_home

# Optimization Algorithms
from optimization.cma_es import run_optimization as run_cma_static
from optimization.cma_es_dynamic import run_optimization as run_cma_dynamic
from optimization.differential_evolution import run_de_optimization

def main():
    # 1. Configurazione Argomenti
    parser = argparse.ArgumentParser(description="Football Formation Optimization")
    
    parser.add_argument("--mode", type=str, default="cma_static", 
                        choices=["cma_static", "cma_dynamic", "de"], 
                        help="Optimization algorithm to use")
    
    parser.add_argument("--scenario", type=str, default=None,
                        help="Name of the scenario in ground_truth.json (overrides historical data)")

    parser.add_argument("--phase", type=str, default="Possesso offensivo",
                        choices=["po", "pd", "fd"],
                        help="Seleziona la fase di gioco da ottimizzare")

    args = parser.parse_args()

    phase_home = args.phase

    if phase_home == "po":
        phase_home = "Possesso offensivo"
        phase_away = "Fase difensiva"
    elif phase_home == "pd":
        phase_home = "Possesso difensivo"
        phase_away = "Fase difensiva"
    elif phase_home == "fd":
        phase_home = "Fase difensiva"
        phase_away = "Possesso offensivo"

    # 2. Setup Dati
    data = setup_scenario(phase_home=phase_home, phase_away=phase_away, scenario_name=args.scenario)
    
    scenario_label = args.scenario if args.scenario else "Storico"
    print(f"\n=== RUNNING: {args.mode.upper()} | SCENARIO: {scenario_label} | FASE: {phase_home} ===\n")

    # 3. Esecuzione Algoritmo Scelto
    final_obstacles = None
    best_vec = None
    cost_history = []

    # --- CASO 1: CMA-ES STATICO ---
    if args.mode == "cma_static":
        best_vec, cost_history = run_cma_static(
            initial_guess=data["initial_guess"],
            obstacles=data["obstacles_matrix"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=phase_home
        )
        final_obstacles = data["obstacles_matrix"]

    # --- CASO 2: CMA-ES DINAMICO ---
    elif args.mode == "cma_dynamic":
        best_vec, cost_history = run_cma_dynamic(
            initial_guess=data["initial_guess"],
            initial_away_df=data["df_away_start"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=phase_home
        )
        # Ricalcoliamo la reazione finale per il plot corretto
        # Qui best_vec è un array numpy, dobbiamo convertirlo prima di passarlo a react_away
        final_home_df = flat_to_formation(best_vec, data["starters_home"])
        final_away_df = react_away_to_home(
            final_home_df, 
            data["df_away_start"], 
            data["ball_position"]
        )
        final_obstacles = final_away_df[["x", "y"]].to_numpy()

    # --- CASO 3: DIFFERENTIAL EVOLUTION ---
    elif args.mode == "de":
        best_vec, best_cost, cost_history = run_de_optimization(
            initial_guess=data["initial_guess"],
            initial_away_df=data["df_away_start"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=phase_home
        )
        
        final_home_df = flat_to_formation(best_vec, data["starters_home"])
        final_away_df = react_away_to_home(
            final_home_df, 
            data["df_away_start"], 
            data["ball_position"]
        )
        final_obstacles = final_away_df[["x", "y"]].to_numpy()

    print(f"\n[RESULT] Optimization finished.")

    # 4. Report Dettagliato
    print_fitness_breakdown(
        formation_data=best_vec,
        player_names=data["starters_home"],
        obstacles=final_obstacles,
        ball_pos=data["ball_position"],
        initial_df_ref=data["df_home_start"],
        phase_name=phase_home
    )
    
    # 5. Salvataggio Risultati e Plot
    output_folder = f"results_plots/{args.mode}_{phase_home.replace(' ', '_')}"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_fmt_df = flat_to_formation(best_vec, data["starters_home"])
    # ===============================

    # A. Plot Convergenza
    plot_convergence(
        cost_history, 
        os.path.join(output_folder, f"convergence_{timestamp}.png")
    )

    # B. Plot Orizzontale (Passiamo il DataFrame)
    plot_formation_with_ball_and_obstacles(
        best_fmt_df, 
        f"Optimized Formation - {phase_home}",
        team='Home',
        color='blue',
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
    )

    # C. Plot Verticale (Passiamo il DataFrame)
    plot_formation_vertical(
        best_fmt_df,
        f"{args.mode.upper()} Optimized - {phase_home}",
        team="Home",
        color="blue",
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
        save_path=os.path.join(output_folder, f"formation_{timestamp}.png")
    )
    
    print(f"Grafici salvati in: {output_folder}")

if __name__ == "__main__":
    main()
    create_evolution_gif()