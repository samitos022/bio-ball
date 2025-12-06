import config
from utils.conversion import flat_to_formation
from utils.away_reaction import react_away_to_home
from optimization.constraints import penalty_total
from optimization.cost_functions import (
    cost_coverage, cost_offside, cost_passing_lanes, cost_ball_support
)

def calculate_total_cost(df_candidate, obstacles_array, ball_pos, initial_df_ref):
    """Calcola il costo totale dato un candidato e degli ostacoli (già calcolati)."""
    
    # 1. Constraints
    pos_dict_for_constraints = {
        "Start": initial_df_ref,
        "Candidate": df_candidate
    }
    c_constraints = penalty_total(pos_dict_for_constraints)
    if c_constraints > config.PENALTY_MAX_THRESHOLD:
        return c_constraints * config.OBJ_W_CONSTRAINTS

    # 2. Objectives
    c_cover    = cost_coverage(df_candidate)
    c_pass     = cost_passing_lanes(df_candidate, obstacles_array, ball_pos)
    c_ball     = cost_ball_support(df_candidate, ball_pos)
    c_offside  = cost_offside(df_candidate, obstacles_array, ball_pos)

    total_cost = (
        config.OBJ_W_CONSTRAINTS * c_constraints +
        config.OBJ_W_COVER       * c_cover +
        config.OBJ_W_PASS        * c_pass +
        config.OBJ_W_BALL        * c_ball +
        config.OBJ_W_OFFSIDE     * c_offside
    )
    return total_cost

def objective_function(vector, args):
    """
    Funzione obiettivo unificata.
    args può essere:
    - (player_names, obstacles_array, ball_pos, initial_df_ref) -> STATIC
    - (player_names, initial_away_df, ball_pos, initial_df_ref, 'dynamic') -> DYNAMIC
    """
    
    player_names = args[0]
    ball_pos = args[2]
    initial_df_ref = args[3]
    
    # Ricostruzione formazione
    df_candidate = flat_to_formation(vector, player_names)
    
    # Gestione Dinamica vs Statica
    if len(args) == 5 and args[4] == 'dynamic':
        # Caso Dinamico: args[1] è il DataFrame iniziale della difesa
        initial_away_df = args[1]
        away_reactive_df = react_away_to_home(
            home_df=df_candidate,
            base_away_df=initial_away_df,
            ball_pos=ball_pos
        )
        obstacles_array = away_reactive_df[["x", "y"]].to_numpy()
    else:
        # Caso Statico: args[1] è già l'array degli ostacoli
        obstacles_array = args[1]

    return calculate_total_cost(df_candidate, obstacles_array, ball_pos, initial_df_ref)