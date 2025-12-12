import pandas as pd
import numpy as np
import json
import os
from utils.load_data import load_and_clean_metrica_tracking, load_match
from utils.analysis_dynamic import average_positions, starters, prepare_obstacles
from utils.initial_pop import average_ball_positions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'metrica', 'sample_game_1')

def setup_scenario(game_dir=DEFAULT_DATA_DIR, 
                   phase_home="Defensive phase", 
                   phase_away="Attacking possession",
                   scenario_name=None):
    """
    Loads the initial data. If scenario_name is specified, it overrides
    the positions of ball and away team according to the Ground Truth
    """
    print("=== LOADING DATA ===")
    tracking_home = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match(f'{game_dir}/Sample_Game_1_RawEventsData.csv')

    print("=== PREPARING SCENARIO ===")
    starters_home = starters(tracking_home, n_players=11)
    starters_away = starters(tracking_away, n_players=11)

    # 1. Default data computing (Historic)
    avg_pos_home = average_positions(match, tracking_home, 'Home')
    avg_pos_away = average_positions(match, tracking_away, 'Away')
    ball_home = average_ball_positions(tracking_home, 'Home')

    ball_position = ball_home[phase_home]
    
    df_home_start = avg_pos_home[phase_home].loc[starters_home]
    initial_guess = df_home_start[['x', 'y']].values.flatten()
    
    # Away Start & Obstacles (Default)
    df_away_start = avg_pos_away[phase_away].loc[starters_away]
    obstacles_matrix = prepare_obstacles(avg_pos_away, phase_away, starters_away)

    # 2. OVERRIDE USING SCENARIO JSON
    if scenario_name:
        json_path = "code/data/formations/ground_truth.json"
        try:
            with open(json_path, "r") as f:
                full_db = json.load(f)
            
            if scenario_name in full_db:
                print(f"✅ SCENARIO OVERRIDE: Loading '{scenario_name}'...")
                scen_data = full_db[scenario_name]
                
                # A. Override Ball
                if "ball" in scen_data and len(scen_data["ball"]) == 2:
                    ball_position = np.array(scen_data["ball"])
                    print(f"   -> New Ball position: {ball_position}")

                # B. Override Avversari (Away)
                if "away" in scen_data:
                    away_dict = scen_data["away"]
                    new_away_coords = []
                    
                    new_away_data = []

                    for i in range(len(starters_away)):
                        idx_str = str(i)
                        if idx_str in away_dict:
                            pos = away_dict[idx_str]
                            new_away_coords.append(pos)
                            new_away_data.append(pos)
                        else:
                            fallback = df_away_start.iloc[i][['x', 'y']].values
                            new_away_coords.append(fallback)
                            new_away_data.append(fallback)
                    
                    obstacles_matrix = np.array(new_away_coords)
                    
                    df_away_start = pd.DataFrame(
                        new_away_data, 
                        index=starters_away, 
                        columns=['x', 'y']
                    )
                    print(f"   -> New Away team loaded: {len(obstacles_matrix)}")

            else:
                print(f"⚠️ Scenario '{scenario_name}' not found. Using default data.")
        
        except FileNotFoundError:
            print(f"⚠️ ERROR: File {json_path} not found. Using default data.")
        except Exception as e:
            print(f"⚠️ ERROR reading JSON: {e}")

    return {
        "starters_home": starters_home,
        "starters_away": starters_away,
        "ball_position": ball_position,
        "initial_guess": initial_guess,
        "df_home_start": df_home_start,
        "df_away_start": df_away_start,
        "obstacles_matrix": obstacles_matrix
    }