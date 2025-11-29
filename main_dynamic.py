from initial_pop import average_ball_positions
from utils.animation_dynamic import create_evolution_gif
from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis_dynamic import average_positions, starters, prepare_obstacles, plot_formation_with_ball_and_obstacles, plot_convergence
from optimization.cma_es_dynamic import run_optimization

def main(): 
    print("Caricamento dati...")
    tracking_home = load_and_clean_metrica_tracking(
        'data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv'
    )
    tracking_away = load_and_clean_metrica_tracking(
        'data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv'
    )
    match = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')

    # --- Starter ---
    starters_home_list = starters(tracking_home, n_players=11)
    starters_away_list = starters(tracking_away, n_players=11)

    print("Analisi posizioni medie storiche...")
    avg_pos_home_dict = average_positions(match, tracking_home, 'Home')
    avg_pos_away_dict = average_positions(match, tracking_away, 'Away')

    print("Calcolo posizioni della palla per fase...")
    ball_home_dict = average_ball_positions(tracking_home, 'Home')

    # --- SCENARIO DI GIOCO ---
    phase_home = "Fase difensiva"
    phase_away = "Possesso offensivo"

    ball_position = ball_home_dict[phase_home]

    # --- AWAY BASE PER MODELLO REATTIVO ---
    initial_away_df = avg_pos_away_dict[phase_away].loc[starters_away_list]

    # --- START HOME POSITION ---
    df_home_start = avg_pos_home_dict[phase_home].loc[starters_home_list]
    initial_guess = df_home_start[['x', 'y']].values.flatten()

    # --- CMA-ES CON AWAY REATTIVA ---
    initial_away_df = avg_pos_away_dict[phase_away].loc[starters_away_list]

    best_solution, cost_history = run_optimization(
        initial_guess=initial_guess,
        initial_away_df=initial_away_df,
        ball_position=ball_position,
        player_names=starters_home_list
    )

    # --- PLOT ---
    plot_convergence(cost_history)

    plot_formation_with_ball_and_obstacles(
        best_solution,
        f"Formazione Ottimizzata - {phase_home}",
        team='Home',
        color='blue',
        ball_position=ball_position,
        obstacles=initial_away_df   # puoi anche usare away reattiva ricostruita
    )


if __name__ == "__main__":
    main()
    create_evolution_gif()