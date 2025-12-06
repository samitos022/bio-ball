import argparse
import os
from datetime import datetime
from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from utils.analysis_dynamic import plot_convergence, plot_formation_vertical, plot_formation_with_ball_and_obstacles
from utils.animation import create_evolution_gif

from optimization.cma_es import run_optimization as run_cma_static
from optimization.cma_es_dynamic import run_optimization as run_cma_dynamic
from optimization.differential_evolution import run_de_optimization

def main():
    parser = argparse.ArgumentParser(description="Football Formation Optimization")
    parser.add_argument("--mode", type=str, default="cma_static", 
                        choices=["cma_static", "cma_dynamic", "de"], 
                        help="Optimization algorithm to use")
    args = parser.parse_args()

    # 1. Setup Dati
    data = setup_scenario()
    
    phase_home = "Fase difensiva" # Solo per naming nei plot
    print(f"=== RUNNING: {args.mode.upper()} ===")

    # 2. Selezione Algoritmo
    if args.mode == "cma_static":
        best_vec, cost_history = run_cma_static(
            initial_guess=data["initial_guess"],
            obstacles=data["obstacles_matrix"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"]
        )
        final_obstacles = data["obstacles_matrix"]

    elif args.mode == "cma_dynamic":
        best_vec, cost_history = run_cma_dynamic(
            initial_guess=data["initial_guess"],
            initial_away_df=data["df_away_start"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"]
        )
        # Nota: in dinamico, gli ostacoli finali dipendono dalla soluzione trovata.
        # Per il plot finale statico usiamo la base (TODO: ricalcolare la reazione)
        final_obstacles = data["obstacles_matrix"] 

    elif args.mode == "de":
        best_vec, best_cost, cost_history = run_de_optimization(
            initial_guess=data["initial_guess"],
            initial_away_df=data["df_away_start"],
            ball_position=data["ball_position"],
            player_names=data["starters_home"]
        )
        final_obstacles = data["obstacles_matrix"]

    # 3. Risultati e Plotting
    print(f"[RESULT] Optimization finished.")
    
    output_folder = f"results_plots/{args.mode}_{phase_home}"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot Convergenza
    plot_convergence(
        cost_history, 
        os.path.join(output_folder, f"convergence_{timestamp}.png")
    )

    plot_formation_with_ball_and_obstacles(
        best_vec,
        f"Formazione Ottimizzata - {phase_home}",
        team='Home',
        color='blue',
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
    )

    # Plot Formazione Verticale
    plot_formation_vertical(
        best_vec,
        f"{args.mode.upper()} Optimized - {phase_home}",
        team="Home",
        color="blue",
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
        save_path=os.path.join(output_folder, f"formation_{timestamp}.png")
    )
    
    print(f"Results saved in {output_folder}")

if __name__ == "__main__":
    main()
    create_evolution_gif()