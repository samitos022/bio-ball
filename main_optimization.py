from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis import average_positions, starters, prepare_obstacles, plot_formation
from optimization.cma_es import run_optimization


def main(): 
    tracking_home = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')

    starters_home_list = starters(tracking_home, n_players=11)
    starters_away_list = starters(tracking_away, n_players=11)

    avg_pos_home_dict = average_positions(match, tracking_home, 'Home')
    avg_pos_away_dict = average_positions(match, tracking_away, 'Away')

    ball_position = (0.6, 0.5)

    if ball_position[0] < 0.5:
        phase_home = "Possesso difensivo"
        phase_away = "Fase difensiva"
        target_team = "Home"
    else:
        phase_home = "Possesso offensivo"
        phase_away = "Fase difensiva"
        target_team = "Home"
    
    obstacles_array = prepare_obstacles(
        avg_pos_away_dict, 
        phase=phase_away, 
        starters_list=starters_away_list
    )

    df_home_start = avg_pos_home_dict[phase_home].loc[starters_home_list]

    initial_guess = df_home_start[['x', 'y']].values.flatten()

    best_solution = run_optimization(
        initial_guess=initial_guess,
        obstacles=obstacles_array,
        ball_position=ball_position,
        player_names=starters_home_list
    )

    plot_formation(best_solution, "Best solution")
    


