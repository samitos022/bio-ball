import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from utils.conversion import flat_to_formation
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
  
def cost_offside(df_home, obstacles_array, ball_pos, attacking_direction='right'):
    home_x = df_home['x'].values
    opp_x = obstacles_array[:, 0]
    
    if attacking_direction == 'right':
        sorted_opp = np.sort(opp_x)
        offside_line = sorted_opp[-2] 
        
        # (Usiamo la palla perché se la palla è oltre la linea, non c'è fuorigioco)
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

def cost_passing_lanes(df_home, obstacles_array, ball_pos, block_threshold=0.03):
    home = df_home[['x', 'y']].values
    n = len(home)

    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier = np.argmin(dists) # indice del passatore
    p1 = home[ball_carrier]

    penalty = 0.0

    w_block = 8.0        # passaggio bloccato = molto penalizzante
    w_long = 1.5         # passaggi lunghi
    w_angle = 0.5        # angolo
    max_pass_len = 0.35  # oltre ~35 metri penalizzo

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

    # Se nessun passaggio è “buono” → penalità extra
    if penalty == 0:
        penalty += 10.0

    return penalty

def cost_ball_support(df, ball_pos):
    """Penalità se nessuno è vicino alla palla"""
    dists = np.linalg.norm(df[['x', 'y']].values - ball_pos, axis=1)
    # Penalizza la distanza del giocatore più vicino alla palla
    return np.min(dists) * 5.0

def objective_function(vector, args):
    player_names, obstacles_array, ball_pos, initial_df_ref = args

    df_candidate = flat_to_formation(vector, player_names)

    pos_dict_for_constraints = {
        "Start": initial_df_ref,
        "Candidate": df_candidate
    }
    c_constraints = penalty_total(pos_dict_for_constraints)

    if c_constraints > 5000:
        return c_constraints

    c_cover    = cost_coverage(df_candidate)
    c_pass     = cost_passing_lanes(df_candidate, obstacles_array, ball_pos)
    c_ball     = cost_ball_support(df_candidate, ball_pos)
    c_offside  = cost_offside(df_candidate, obstacles_array, ball_pos)

    w_constraints = 1.0
    w_cover       = 1.0
    w_pass        = 1.0
    w_ball        = 20.0
    w_offside     = 100.0

    total_cost = (
        w_constraints * c_constraints +
        w_cover       * c_cover +
        w_pass        * c_pass +
        w_ball        * c_ball +
        w_offside     * c_offside
    )

    return total_cost