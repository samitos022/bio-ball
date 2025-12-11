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
                   phase_home="Fase difensiva", 
                   phase_away="Possesso offensivo",
                   scenario_name=None):
    """
    Carica i dati. Se scenario_name è specificato, sovrascrive
    la posizione della palla e degli avversari con quelli del Ground Truth JSON.
    """
    print("=== LOADING DATA ===")
    tracking_home = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking(f'{game_dir}/Sample_Game_1_RawTrackingData_Away_Team.csv')
    match = load_match(f'{game_dir}/Sample_Game_1_RawEventsData.csv')

    print("=== PREPARING SCENARIO ===")
    starters_home = starters(tracking_home, n_players=11)
    starters_away = starters(tracking_away, n_players=11)

    # 1. Calcolo dati storici (Base)
    avg_pos_home = average_positions(match, tracking_home, 'Home')
    avg_pos_away = average_positions(match, tracking_away, 'Away')
    ball_home = average_ball_positions(tracking_home, 'Home')

    # Dati di default (Storici)
    ball_position = ball_home[phase_home]
    
    # Home Start (Sempre basato sulla media storica come punto di partenza neutro)
    df_home_start = avg_pos_home[phase_home].loc[starters_home]
    initial_guess = df_home_start[['x', 'y']].values.flatten()
    
    # Away Start & Obstacles (Default)
    df_away_start = avg_pos_away[phase_away].loc[starters_away]
    obstacles_matrix = prepare_obstacles(avg_pos_away, phase_away, starters_away)

    # 2. OVERRIDE CON SCENARIO JSON (Se richiesto)
    if scenario_name:
        json_path = "code/data/formations/ground_truth.json"
        try:
            with open(json_path, "r") as f:
                full_db = json.load(f)
            
            if scenario_name in full_db:
                print(f"✅ SCENARIO OVERRIDE: Caricamento '{scenario_name}'...")
                scen_data = full_db[scenario_name]
                
                # A. Override Palla
                if "ball" in scen_data and len(scen_data["ball"]) == 2:
                    ball_position = np.array(scen_data["ball"])
                    print(f"   -> Nuova Palla: {ball_position}")

                # B. Override Avversari (Away)
                if "away" in scen_data:
                    away_dict = scen_data["away"]
                    # Costruiamo una lista ordinata di posizioni
                    # Assumiamo che gli indici nel JSON '0', '1'... corrispondano 
                    # all'ordine in starters_away
                    new_away_coords = []
                    
                    # Creiamo un DataFrame temporaneo per df_away_start
                    new_away_data = []

                    for i in range(len(starters_away)):
                        idx_str = str(i)
                        if idx_str in away_dict:
                            pos = away_dict[idx_str]
                            new_away_coords.append(pos)
                            new_away_data.append(pos)
                        else:
                            # Fallback se manca un giocatore nel JSON
                            # Usiamo la posizione media storica per quel giocatore
                            fallback = df_away_start.iloc[i][['x', 'y']].values
                            new_away_coords.append(fallback)
                            new_away_data.append(fallback)
                    
                    # Aggiorniamo obstacles_matrix (Numpy array)
                    obstacles_matrix = np.array(new_away_coords)
                    
                    # Aggiorniamo df_away_start (Pandas DataFrame) per il modo dinamico
                    df_away_start = pd.DataFrame(
                        new_away_data, 
                        index=starters_away, 
                        columns=['x', 'y']
                    )
                    print(f"   -> Nuovi Avversari caricati: {len(obstacles_matrix)}")

            else:
                print(f"⚠️ ATTENZIONE: Scenario '{scenario_name}' non trovato nel JSON. Uso dati storici.")
        
        except FileNotFoundError:
            print(f"⚠️ ERRORE: File {json_path} non trovato. Uso dati storici.")
        except Exception as e:
            print(f"⚠️ ERRORE lettura JSON: {e}")

    return {
        "starters_home": starters_home,
        "starters_away": starters_away,
        "ball_position": ball_position,
        "initial_guess": initial_guess,
        "df_home_start": df_home_start,
        "df_away_start": df_away_start,
        "obstacles_matrix": obstacles_matrix
    }