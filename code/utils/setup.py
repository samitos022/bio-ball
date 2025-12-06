import pandas as pd
from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis_dynamic import average_positions, starters, prepare_obstacles
from utils.initial_pop import average_ball_positions

def setup_scenario(game_dir='code/data/metrica/sample_game_1', 
                   phase_home="Fase difensiva", 
                   phase_away="Possesso offensivo"):
    """
    Carica i dati, calcola le medie e prepara lo scenario iniziale.
    """
    print("=== LOADING DATA ===")
    tracking_home = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match(f'{game_dir}/Sample_Game_1_RawEventsData.csv')

    print("=== PREPARING SCENARIO ===")
    starters_home = starters(tracking_home, n_players=11)
    starters_away = starters(tracking_away, n_players=11)

    avg_pos_home = average_positions(match, tracking_home, 'Home')
    avg_pos_away = average_positions(match, tracking_away, 'Away')
    ball_home = average_ball_positions(tracking_home, 'Home')

    ball_position = ball_home[phase_home]
    
    # Home Start
    df_home_start = avg_pos_home[phase_home].loc[starters_home]
    initial_guess = df_home_start[['x', 'y']].values.flatten()
    
    # Away Start (Base for dynamic or static)
    df_away_start = avg_pos_away[phase_away].loc[starters_away]
    
    # Obstacles array (Static version)
    obstacles_matrix = prepare_obstacles(avg_pos_away, phase_away, starters_away)

    return {
        "starters_home": starters_home,
        "starters_away": starters_away,
        "ball_position": ball_position,
        "initial_guess": initial_guess,
        "df_home_start": df_home_start,
        "df_away_start": df_away_start,
        "obstacles_matrix": obstacles_matrix
    }