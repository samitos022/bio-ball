import argparse
import os
from datetime import datetime

from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from utils.analysis_dynamic import plot_convergence, plot_formation_with_ball_and_obstacles
from utils.away_reaction import react_away_to_home

# Importiamo il nuovo algoritmo GECCO
from optimization.cma_es_gecco import run_optimization_gecco

def main():
    parser = argparse.ArgumentParser(description="Football Formation Optimization - GECCO VERSION")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--phase", type=str, default="Attacking possession")
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

    data = setup_scenario(phase_home=phase_home, phase_away=phase_away, scenario_name=args.scenario)
    
    print(f"\n=== RUNNING: GECCO PIPELINE | SCENARIO: {args.scenario} | FASE: {phase_home} ===\n")

    # Eseguiamo solo il nuovo CMA
    best_vec, cost_history = run_optimization_gecco(
        initial_guess=data["initial_guess"],
        initial_away_df=data["df_away_start"],
        ball_position=data["ball_position"],
        player_names=data["starters_home"],
        phase_name=phase_home
    )

    # Calcoliamo la posizione finale degli avversari per il plot
    final_home_df = flat_to_formation(best_vec, data["starters_home"])
    final_away_df = react_away_to_home(final_home_df, data["df_away_start"], data["ball_position"])
    final_obstacles = final_away_df[["x", "y"]].to_numpy()

    print("\n[RESULT] Optimization finished.")

    # Salvataggio Plot
    output_folder = f"results_gecco/{phase_home.replace(' ', '_')}"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_convergence(cost_history, os.path.join(output_folder, f"convergence_{timestamp}.png"))

    plot_formation_with_ball_and_obstacles(
        final_home_df, 
        f"GECCO Opt. Formation - {phase_home}",
        team='Home', color='blue',
        ball_position=data["ball_position"],
        obstacles=final_obstacles,
    )
    
    print(f"Graphs stored in: {output_folder}")

if __name__ == "__main__":
    main()