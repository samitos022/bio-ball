import numpy as np
import config_gecco as cfg
from utils.conversion import flat_to_formation
from optimization.constraints_gecco import calculate_penalties
from utils.spatial_math import generate_xt_map, calculate_pitch_control

XT_MAP_ATTACK = generate_xt_map(phase="Attacking")
XT_MAP_DEFENSE = generate_xt_map(phase="Defending")

# Calcolo dell'area di una cella per normalizzare le somme spaziali
CELL_AREA = 1.0 / (cfg.GRID_RES ** 2)

def objective_function_gecco(vector, args):
    detailed = False
    if len(args) == 7:
        player_names, obstacles_array, ball_pos, initial_df_ref, phase_name, mode, detailed = args
    else:
        player_names, obstacles_array, ball_pos, initial_df_ref, phase_name, mode, = args

    df_candidate = flat_to_formation(vector, player_names)
    home_coords = df_candidate[['x', 'y']].values
    
    
    away_coords = obstacles_array

    # 1. PENALITÀ
    penalties_data = calculate_penalties(
        home_coords=home_coords, away_coords=away_coords, 
        ball_pos=ball_pos, phase_name=phase_name, detailed=detailed
    )
    c_hard_total = penalties_data["Total"] if detailed else penalties_data

    # 2. GHOSTING
    ghost_coords = initial_df_ref[['x', 'y']].values
    distances = np.linalg.norm(home_coords - ghost_coords, axis=1)
    dist_penalized = np.maximum(0, distances - 0.04) 
    ghost_cost = np.sum(dist_penalized**2)
    ghosting_penalty_scaled = cfg.LAMBDA_GHOSTING * ghost_cost

    # 3. VALORE TATTICO
    pc_matrix_home = calculate_pitch_control(home_coords, away_coords, ball_pos)    
    pc_matrix_away = 1.0 - pc_matrix_home 
    
    is_attack = ("Attacking" in phase_name or "possession" in phase_name)  

    attack_reward_scaled = 0.0
    counterattack_risk_scaled = 0.0
    defensive_reward_scaled = 0.0
    
    if is_attack:
        # FASE OFFENSIVA O POSSESSO DIFENSIVO (Normalizzata con CELL_AREA)
        raw_attack = np.sum(pc_matrix_home * XT_MAP_ATTACK) * CELL_AREA
        raw_risk = np.sum(pc_matrix_away * XT_MAP_DEFENSE) * CELL_AREA
        
        attack_reward_scaled = raw_attack * getattr(cfg, 'WEIGHT_TACTICAL_ATTACK', 1.0)
        counterattack_risk_scaled = raw_risk * getattr(cfg, 'WEIGHT_REST_DEFENSE', 3.0)
        
        fitness = -attack_reward_scaled + counterattack_risk_scaled + ghosting_penalty_scaled + c_hard_total
    else:
        # FASE DIFENSIVA (Normalizzata con CELL_AREA)
        raw_defense = np.sum(pc_matrix_home * XT_MAP_DEFENSE) * CELL_AREA
        defensive_reward_scaled = raw_defense * getattr(cfg, 'WEIGHT_TACTICAL_DEFEND', 1.0)
        
        fitness = -defensive_reward_scaled + ghosting_penalty_scaled + c_hard_total

    if detailed:
        return {
            "FITNESS": fitness,
            "Attack_Reward": -attack_reward_scaled if is_attack else 0.0, 
            "Counter_Risk": counterattack_risk_scaled,
            "Def_Reward": -defensive_reward_scaled if not is_attack else 0.0,
            "Ghost_Penalty": ghosting_penalty_scaled,
            "Hard_Penalties": penalties_data
        }

    return fitness