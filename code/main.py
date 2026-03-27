import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import time

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
    # 1. Arguments configuration
    parser = argparse.ArgumentParser(description="Football Formation Optimization")
    
    parser.add_argument("--mode", type=str, default="cma_static", 
                        choices=["cma_static", "cma_dynamic", "de"], 
                        help="Optimization algorithm to use")
    
    parser.add_argument("--scenario", type=str, default=None,
                        help="Name of the scenario in ground_truth.json (overrides historical data)")

    parser.add_argument("--phase", type=str, default="pa",
                        choices=["pa", "pd", "dp"],
                        help="Game phase to optimize")

    args = parser.parse_args()

    phase_home = args.phase

    if phase_home == "pa":
        phase_home = "Attacking possession"
        phase_away = "Defensive phase"
    elif phase_home == "pd":
        phase_home = "Defensive possession"
        phase_away = "Defensive phase"
    elif phase_home == "dp":
        phase_home = "Defensive phase"
        phase_away = "Attacking possession"

    # 2. Setup
    data = setup_scenario(phase_home=phase_home, phase_away=phase_away, scenario_name=args.scenario)
    
    scenario_label = args.scenario if args.scenario else "Metrica"
    print(f"\n=== RUNNING: {args.mode.upper()} | SCENARIO: {scenario_label} | FASE: {phase_home} ===\n")

    # 3. Execution
    final_obstacles = None
    best_vec = None
    cost_history = []

    start_time = time.perf_counter()

    # --- 1: CMA-ES STATIC ---
    if args.mode == "cma_static":
        best_vec, cost_history = run_cma_static(
            initial_guess=data["initial_guess"],
            obstacles=data["obstacles_matrix"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"],
            phase_name=phase_home
        )
        final_obstacles = data["obstacles_matrix"]

    # --- 2: CMA-ES DYNAMIC ---
    elif args.mode == "cma_dynamic":
        best_vec, cost_history = run_cma_dynamic(
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

    # --- 3: DIFFERENTIAL EVOLUTION ---
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

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"\n[RESULT] Optimization finished.")

    # 4. Detailed Report (Catturiamo le metriche nel dizionario)
    fitness_metrics = print_fitness_breakdown(
        formation_data=best_vec,
        player_names=data["starters_home"],
        obstacles=final_obstacles,
        ball_pos=data["ball_position"],
        initial_df_ref=data["df_home_start"],
        phase_name=phase_home
    )
    
    # 5. Salvataggio Organizzato per il Paper
    output_folder = f"results_plots/{args.mode}_{phase_home.replace(' ', '_')}"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_fmt_df = flat_to_formation(best_vec, data["starters_home"])

    # A. FILE SINGOLO DEL TEST (Posizioni + Metriche Principali)
    # Creiamo una copia del dataframe delle posizioni
    run_report_df = best_fmt_df.copy()
    
    # Aggiungiamo le metriche come nuove colonne (si ripeteranno per ogni riga, è normale nei CSV)
    run_report_df['Total_Fitness'] = fitness_metrics.get("Total_Fitness", 0)
    
    # Filtriamo per salvare solo i valori "Raw" degli obiettivi (più utili per l'analisi)
    for key, value in fitness_metrics.items():
        if "_Raw" in key:
            run_report_df[key] = value

    csv_run_path = os.path.join(output_folder, f"run_report_{timestamp}.csv")
    run_report_df.to_csv(csv_run_path, index_label="Player")

    # B. MASTER LOG (Meta-dati + Fitness + Coordinate orizzontali)
    master_log_path = "results_plots/master_log_experiments.csv"
    
    # Costruiamo la riga base
    log_row = {
        "Timestamp": timestamp,
        "Algorithm": args.mode,
        "Scenario": scenario_label,
        "Phase": phase_home,
        "Execution_Time_sec": round(execution_time, 3),
        "Total_Fitness": fitness_metrics.get("Total_Fitness", 0)
    }
    
    # Aggiungiamo le coordinate X e Y di OGNI giocatore come nuove colonne
    # Es: Player1_X, Player1_Y, Player2_X...
    for player_name, row in best_fmt_df.iterrows():
        log_row[f"{player_name}_X"] = row['x']
        log_row[f"{player_name}_Y"] = row['y']
        
    log_df = pd.DataFrame([log_row])
    
    # Salvataggio robusto: se il file esiste, aggiungiamo in coda
    if not os.path.isfile(master_log_path):
        log_df.to_csv(master_log_path, index=False)
    else:
        log_df.to_csv(master_log_path, mode='a', header=False, index=False)

    # 6. PLOTS

    # A. Convergence Plot
    plot_convergence(
        cost_history, 
        os.path.join(output_folder, f"convergence_{timestamp}.pdf")
    )

    # B. Plot Orizzontale
    plot_formation_with_ball_and_obstacles(
        best_fmt_df, 
        f"Optimized Formation - {phase_home}",
        team='Home',
        color='blue',
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
    )

    # C. Plot Verticale
    plot_formation_vertical(
        best_fmt_df,
        f"{args.mode.upper()} Optimized Formation - {phase_home}",
        team="Home",
        color="blue",
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
        save_path=os.path.join(output_folder, f"formation_{timestamp}.pdf")
    )
    
    print(f"File salvati: {csv_run_path}")
    print(f"Master Log aggiornato in: {master_log_path}")

if __name__ == "__main__":
    main()
    # (Opzionale: togli create_evolution_gif() se non ti serve più l'animazione)
    create_evolution_gif()