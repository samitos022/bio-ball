from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis import average_positions, starters, prepare_obstacles, plot_formation_with_ball_and_obstacles, plot_convergence
from optimization.cma_es import run_optimization

def main(): 
    print("Caricamento dati...")
    tracking_home = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')

    starters_home_list = starters(tracking_home, n_players=11)
    starters_away_list = starters(tracking_away, n_players=11)

    print("Analisi posizioni medie storiche...")
    avg_pos_home_dict = average_positions(match, tracking_home, 'Home')
    avg_pos_away_dict = average_positions(match, tracking_away, 'Away')

    # SCENARIO
    ball_position = (0.25, 0.5) # Palla a sinistra
    print(f"Scenario: Palla in {ball_position}")

    if ball_position[0] < 0.5:
        phase_home = "Possesso difensivo"
        phase_away = "Fase difensiva"
    else:
        phase_home = "Possesso offensivo"
        phase_away = "Fase difensiva"
    
    # Preparazione Ostacoli (Avversari Statici)
    obstacles_array = prepare_obstacles(
        avg_pos_away_dict, 
        phase=phase_away, 
        starters_list=starters_away_list
    )

    # Preparazione Squadra Target (Punto di partenza)
    df_home_start = avg_pos_home_dict[phase_home].loc[starters_home_list]
    initial_guess = df_home_start[['x', 'y']].values.flatten()

    # ESECUZIONE OTTIMIZZAZIONE
    best_solution, cost_history = run_optimization(
        initial_guess=initial_guess,
        obstacles=obstacles_array,
        ball_position=ball_position,
        player_names=starters_home_list
    )

    plot_convergence(cost_history)

    plot_formation_with_ball_and_obstacles(best_solution, f"Formazione Ottimizzata - {phase_home}", team='Home', color='blue', ball_position=ball_position, obstacles=obstacles_array)

if __name__ == "__main__":
    main()