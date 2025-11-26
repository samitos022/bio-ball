import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from utils.conversion import flat_to_formation

def penalty_total(positions, field_limits=(1.0, 1.0), min_dist=0.05, 
                  transition_weight=1.0, boundary_weight=5.0, proximity_weight=2.0):
    penalty = 0.0
    phases = list(positions.keys())

    # A. Penalità Confini e Prossimità (Collisioni)
    for phase_name, df in positions.items():
        pos = df[["x", "y"]].to_numpy()

        # Boundaries
        out_of_bounds_x = np.sum((pos[:, 0] < 0) | (pos[:, 0] > field_limits[0]))
        out_of_bounds_y = np.sum((pos[:, 1] < 0) | (pos[:, 1] > field_limits[1]))
        penalty += boundary_weight * (out_of_bounds_x + out_of_bounds_y)

        n_players = len(pos)
        
        # Proximity (Collisioni tra giocatori)
        for i in range(n_players):
            for j in range(i + 1, n_players):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_dist:
                    penalty += proximity_weight * (min_dist - dist) ** 2

    # B. Penalità Transizione (Movimento rispetto alla fase precedente)
    for i in range(len(phases) - 1):
        df1 = positions[phases[i]]
        df2 = positions[phases[i + 1]]

        common_players = df1.index.intersection(df2.index)
        p1 = df1.loc[common_players, ["x", "y"]].to_numpy()
        p2 = df2.loc[common_players, ["x", "y"]].to_numpy()

        diffs = np.linalg.norm(p2 - p1, axis=1)
        penalty += transition_weight * np.sum(diffs ** 2)

    return penalty


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

def cost_passing_lanes(df_home, obstacles_array, block_threshold=0.03):
    home = df_home[['x', 'y']].values
    n = len(home)
    penalty = 0.0
    
    w_block = 5.0      # Peso linea bloccata
    w_long_pass = 1.0  # Peso passaggi troppo lunghi
    max_pass_len = 0.4 # Oltre 40 metri iniziamo a penalizzare
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            p1, p2 = home[i], home[j]
            
            dist = np.linalg.norm(p1 - p2)
            
            dist_penalty = 0.0
            if dist > max_pass_len:
                dist_penalty = w_long_pass * (dist - max_pass_len)**2
            
            # Angolo (preferiamo sempre passaggi in avanti/aperti)
            # Nota: riduciamo un po' l'impatto dell'angolo per evitare distorsioni
            angle_pen = 0.2 * (1 - angle_score(p1, p2))

            # Blocco Ostacoli
            block = False
            if dist > 0.01: 
                for opp in obstacles_array:
                    if point_line_distance(opp, p1, p2) < block_threshold:
                        block = True
                        break
            
            penalty += dist_penalty + angle_pen + (w_block if block else 0.0)
            
    return penalty

def cost_ball_support(df, ball_pos):
    """Penalità se nessuno supporta la palla (troppo lontani)."""
    dists = np.linalg.norm(df[['x', 'y']].values - ball_pos, axis=1)
    # Penalizza la distanza del giocatore più vicino alla palla
    return np.min(dists) * 5.0

# =============================================================================
# 3. FUNZIONE OBIETTIVO (Wrapper per CMA-ES)
# =============================================================================

def objective_function(vector, args):
    player_names, obstacles_array, ball_pos, initial_df_ref = args
    
    # 1. Decoding
    df_candidate = flat_to_formation(vector, player_names)
    
    # 2. Constraints (Collega)
    pos_dict_for_penalty = {
        "Start": initial_df_ref, 
        "Candidate": df_candidate
    }
    
    c_constraints = penalty_total(
        pos_dict_for_penalty,
        field_limits=(1.0, 1.0),
        min_dist=0.06,
        boundary_weight=1000.0, 
        transition_weight=1.0,
        proximity_weight=500.0
    )
    
    if c_constraints > 5000: return c_constraints

    c_cover = cost_coverage(df_candidate)
    c_pass = cost_passing_lanes(df_candidate, obstacles_array)
    c_ball = cost_ball_support(df_candidate, ball_pos)
    
    # 4. Pesi
    w_constraints = 1.0 
    w_cover = 7.0       
    w_pass = 1.0  
    w_ball = 2.0         
    
    total_cost = (c_constraints * w_constraints) + \
                 (c_cover * w_cover) + \
                 (c_pass * w_pass) + \
                 (c_ball * w_ball)
                 
    return total_cost