import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from utils.conversion import flat_to_formation
import config 
from optimization.constraints import penalty_total 

def point_line_distance(point, a, b):
    a, b, p = np.array(a), np.array(b), np.array(point)
    if np.all(a == b): return np.linalg.norm(p - a)
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p - (a + t * (b - a)))

def angle_score(p_i, p_j, team_direction=np.array([1.0, 0.0])):
    v = p_j - p_i
    norm = np.linalg.norm(v)
    if norm == 0: return 0.0
    return (np.dot(v/norm, team_direction) + 1) / 2

def cost_coverage(df):
    points = df[['x', 'y']].values
    if len(points) < 3: return 1.0
    try:
        return 1.0 - ConvexHull(points).volume
    except:
        return 1.0
  
def cost_offside(df_home, obstacles_array, ball_pos, attacking_direction=config.OFFSIDE_ATTACK_DIR):
    home_x = df_home['x'].values
    opp_x = obstacles_array[:, 0]
    
    if attacking_direction == 'right':
        sorted_opp = np.sort(opp_x)
        offside_line = sorted_opp[-2] 
        mask_offside = (home_x > offside_line) & (home_x > ball_pos[0])
        
        if np.any(mask_offside):
            return np.sum(home_x[mask_offside] - offside_line)
            
    else:
        sorted_opp = np.sort(opp_x)
        offside_line = sorted_opp[1]
        mask_offside = (home_x < offside_line) & (home_x < ball_pos[0])
        
        if np.any(mask_offside):
            return np.sum(offside_line - home_x[mask_offside])

    return 0.0

def cost_passing_lanes(df_home, obstacles_array, ball_pos, block_threshold=config.PASS_BLOCK_THRESHOLD):
    home = df_home[['x', 'y']].values
    n = len(home)

    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier = np.argmin(dists) 
    p1 = home[ball_carrier]

    penalty = 0.0

    # Usiamo i valori dal config
    w_block = config.PASS_W_BLOCK
    w_long = config.PASS_W_LONG
    w_angle = config.PASS_W_ANGLE
    max_pass_len = config.PASS_MAX_LEN

    for j in range(n):
        if j == ball_carrier:
            continue

        p2 = home[j]
        dist = np.linalg.norm(p1 - p2)

        # Penalità sulla lunghezza
        long_pen = 0.0
        if dist > max_pass_len:
            long_pen = w_long * (dist - max_pass_len)**2

        # Penalità angolo
        ang_pen = w_angle * (1 - angle_score(p1, p2))

        # Penalità blocchi
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < block_threshold:
                block = True
                break

        block_pen = w_block if block else 0.0

        penalty += long_pen + ang_pen + block_pen

    if penalty == 0:
        penalty += config.PASS_PENALTY_NO_OPTS

    return penalty

def cost_ball_support(df, ball_pos):
    dists = np.linalg.norm(df[['x', 'y']].values - ball_pos, axis=1)
    return np.min(dists) * config.BALL_SUPPORT_W_MULT


def objective_function(vector, args):
    player_names, obstacles_array, ball_pos, initial_df_ref = args

    df_candidate = flat_to_formation(vector, player_names)

    pos_dict_for_constraints = {
        "Start": initial_df_ref,
        "Candidate": df_candidate
    }
    
    # penalty_total ora userà i default presi da config se non specificati
    c_constraints = penalty_total(pos_dict_for_constraints)

    if c_constraints > config.PENALTY_MAX_THRESHOLD:
        return c_constraints

    c_cover    = cost_coverage(df_candidate)
    c_pass     = cost_passing_lanes(df_candidate, obstacles_array, ball_pos)
    c_ball     = cost_ball_support(df_candidate, ball_pos)
    c_offside  = cost_offside(df_candidate, obstacles_array, ball_pos)

    # Pesi presi dal config
    total_cost = (
        config.OBJ_W_CONSTRAINTS * c_constraints +
        config.OBJ_W_COVER       * c_cover +
        config.OBJ_W_PASS        * c_pass +
        config.OBJ_W_BALL        * c_ball +
        config.OBJ_W_OFFSIDE     * c_offside
    )

    return total_cost