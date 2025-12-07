import numpy as np
from scipy.spatial import ConvexHull
import config

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

def cost_coverage(df, detailed=False):
    points = df[['x', 'y']].values
    area = 0.0
    if len(points) >= 3:
        try:
            area = ConvexHull(points).volume
        except:
            area = 0.0
    
    cost = 1.0 - area
    
    if detailed:
        return {
            "total": cost,
            "raw_area": area,
            "empty_space": 1.0 - area
        }
    return cost

def cost_offside(df_home, obstacles_array, ball_pos, attacking_direction=config.OFFSIDE_ATTACK_DIR, detailed=False):
    home_x = df_home['x'].values
    opp_x = obstacles_array[:, 0]
    
    offside_dist = 0.0
    num_players_offside = 0
    
    if attacking_direction == 'right':
        if len(opp_x) >= 2:
            sorted_opp = np.sort(opp_x)
            offside_line = sorted_opp[-2] 
            effective_line = max(offside_line, ball_pos[0])
            mask_offside = (home_x > effective_line)
            if np.any(mask_offside):
                offside_dist = np.sum(home_x[mask_offside] - effective_line)
                num_players_offside = np.sum(mask_offside)
    else:
        if len(opp_x) >= 2:
            sorted_opp = np.sort(opp_x)
            offside_line = sorted_opp[1]
            effective_line = min(offside_line, ball_pos[0])
            mask_offside = (home_x < effective_line)
            if np.any(mask_offside):
                offside_dist = np.sum(effective_line - home_x[mask_offside])
                num_players_offside = np.sum(mask_offside)
    
    if detailed:
        return {
            "total": offside_dist,
            "players_offside": num_players_offside,
            "total_meters": offside_dist
        }
    return offside_dist

def cost_passing_lanes(df_home, obstacles_array, ball_pos, detailed=False):
    home = df_home[['x', 'y']].values
    n = len(home)
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier = np.argmin(dists) 
    p1 = home[ball_carrier]

    # Accumulatori per dettagli
    pen_long = 0.0
    pen_angle = 0.0
    pen_block = 0.0
    
    valid_passes = 0
    blocked_passes_count = 0

    for j in range(n):
        if j == ball_carrier: continue
        p2 = home[j]
        dist = np.linalg.norm(p1 - p2)

        # Lunghezza
        if dist > config.PASS_MAX_LEN:
            pen_long += config.PASS_W_LONG * (dist - config.PASS_MAX_LEN)**2

        # Angolo
        pen_angle += config.PASS_W_ANGLE * (1 - angle_score(p1, p2))

        # Blocco
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            pen_block += config.PASS_W_BLOCK
            blocked_passes_count += 1
        else:
            valid_passes += 1

    penalty_no_opts = 0.0
    if valid_passes == 0:
        penalty_no_opts = config.PASS_PENALTY_NO_OPTS

    total_penalty = pen_long + pen_angle + pen_block + penalty_no_opts

    if detailed:
        return {
            "total": total_penalty,
            "p_long": pen_long,
            "p_angle": pen_angle,
            "p_block": pen_block,
            "p_no_options": penalty_no_opts,
            "num_blocked": blocked_passes_count,
            "num_valid": valid_passes
        }
    return total_penalty

def cost_ball_support(df, ball_pos, detailed=False):
    dists = np.linalg.norm(df[['x', 'y']].values - ball_pos, axis=1)
    min_dist = np.min(dists)
    cost = min_dist * config.BALL_SUPPORT_W_MULT
    
    if detailed:
        return {
            "total": cost,
            "min_distance_meters": min_dist
        }
    return cost