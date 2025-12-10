import config
from utils.conversion import flat_to_formation
from utils.away_reaction import react_away_to_home
from optimization.constraints import penalty_total
from optimization.cost_functions import (
    cost_coverage, cost_passing_lanes, cost_offside_avoidance,
    cost_marking, cost_defensive_compactness, cost_defensive_line_height,
    cost_ball_pressure
)

def objective_function(vector, args):
    """
    args expected: (player_names, obstacles, ball_pos, initial_df_ref, phase_name, mode)
    """
    player_names = args[0]
    ball_pos = args[2]
    initial_df_ref = args[3]
    phase_name = args[4]
    mode = args[5] # 'static' or 'dynamic'

    # Ricostruzione
    df_candidate = flat_to_formation(vector, player_names)

    # Gestione Dinamica (ricalcolo ostacoli)
    if mode == 'dynamic':
        initial_away_df = args[1]
        away_reactive = react_away_to_home(df_candidate, initial_away_df, ball_pos)
        obstacles_array = away_reactive[["x", "y"]].to_numpy()
    else:
        obstacles_array = args[1] # Static obstacles

    # 1. Constraints (Sempre attivi)
    pos_dict = {"Start": initial_df_ref, "Candidate": df_candidate}
    c_hard = penalty_total(pos_dict)
    
    # Se violiamo i vincoli duri, usciamo subito
    if c_hard > config.PENALTY_MAX_THRESHOLD:
        return c_hard * config.OBJ_W_CONSTRAINTS

    # 2. Caricamento Pesi Fase
    weights = config.PHASE_WEIGHTS.get(phase_name, config.PHASE_WEIGHTS["Fase difensiva"])
    
    total_cost = c_hard * config.OBJ_W_CONSTRAINTS

    # 3. Calcolo Obiettivi in base alla Fase
    # OFFENSIVE / POSSESSION
    if weights["W_COVERAGE"] > 0:
        total_cost += cost_coverage(df_candidate) * weights["W_COVERAGE"]
    
    if weights["W_PASSING"] > 0:
        total_cost += cost_passing_lanes(df_candidate, obstacles_array, ball_pos, phase_type=phase_name) * weights["W_PASSING"]
        
    if weights["W_OFFSIDE"] > 0:
        total_cost += cost_offside_avoidance(df_candidate, obstacles_array, ball_pos) * weights["W_OFFSIDE"]

    # DEFENSIVE
    if weights["W_MARKING"] > 0:
        total_cost += cost_marking(df_candidate, obstacles_array) * weights["W_MARKING"]
        
    if weights["W_COMPACTNESS"] > 0:
        total_cost += cost_defensive_compactness(df_candidate) * weights["W_COMPACTNESS"]
        
    if weights["W_LINE_HEIGHT"] > 0:
        total_cost += cost_defensive_line_height(df_candidate, ball_pos) * weights["W_LINE_HEIGHT"]

    # COMMON (Ball Pressure / Support)
    if weights["W_BALL_PRESS"] > 0:
        total_cost += cost_ball_pressure(df_candidate, ball_pos) * weights["W_BALL_PRESS"]

    return total_cost