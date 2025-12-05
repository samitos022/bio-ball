import numpy as np
import pandas as pd
from optimization.differential_evolution import run_de_optimization
from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis_dynamic import average_positions, plot_convergence, plot_formation_vertical, starters, prepare_obstacles, plot_formation_with_ball_and_obstacles
from utils.conversion import flat_to_formation
from initial_pop import average_ball_positions


def main():
    print("=== LOADING DATA ===")
    tracking_home = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')

    print("=== STARTERS ===")
    starters_home = starters(tracking_home)
    starters_away = starters(tracking_away)

    print("=== AVERAGE POSITIONS ===")
    avg_pos_home = average_positions(match, tracking_home, "Home")
    avg_pos_away = average_positions(match, tracking_away, "Away")

    print("=== BALL POSITIONS ===")
    ball_home = average_ball_positions(tracking_home, "Home")

    phase_home = "Possesso offensivo"
    phase_away = "Fase difensiva"

    ball_position = ball_home[phase_home]
    
    print(f"[SCENARIO] {phase_home}")

    df_home_start = avg_pos_home[phase_home].loc[starters_home]
    initial_guess = df_home_start[['x', 'y']].to_numpy().flatten()

    away_df_start = avg_pos_away[phase_away].loc[starters_away]

    obstacles_matrix = prepare_obstacles(
        avg_pos_away, phase_away, starters_away
    )

    print("=== RUNNING Differential Evolution ===")

    best_vec, best_cost, cost_history = run_de_optimization(
        initial_guess=initial_guess,
        initial_away_df=away_df_start,
        ball_position=ball_position,
        player_names=starters_home
    )

    print(f"[RESULT] Best DE cost: {best_cost:.4f}")

    df_home_opt = flat_to_formation(best_vec, starters_home)

    print("=== PLOTTING DE RESULT ===")

    plot_formation_with_ball_and_obstacles(
        df_home_opt,
        f"DE Formazione Ottimizzata – {phase_home}",
        team="Home",
        color="blue",
        ball_position=ball_position,
        obstacles=obstacles_matrix
    )

    plot_formation_vertical(
        positions=df_home_opt,
        title=f"DE Formazione Ottimizzata – {phase_home}",
        team="Home",
        color="blue",
        ball_position=ball_position,
        obstacles=obstacles_matrix,
        save_path="plot_de/de_formation_vertical.png"
    )

    plot_convergence(
        history=cost_history,
        save_path="plot_de/de_convergence.png"
    )

if __name__ == "__main__":
    main()
