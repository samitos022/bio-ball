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
        if len(opp_x) < 2: return 0.0
        sorted_opp = np.sort(opp_x)
        offside_line = sorted_opp[-2] 
        effective_line = max(offside_line, ball_pos[0])
        mask_offside = (home_x > effective_line)
        if np.any(mask_offside):
            return np.sum(home_x[mask_offside] - effective_line)
    else:
        if len(opp_x) < 2: return 0.0
        sorted_opp = np.sort(opp_x)
        offside_line = sorted_opp[1]
        effective_line = min(offside_line, ball_pos[0])
        mask_offside = (home_x < effective_line)
        if np.any(mask_offside):
            return np.sum(effective_line - home_x[mask_offside])
    return 0.0

def cost_passing_lanes(df_home, obstacles_array, ball_pos):
    home = df_home[['x', 'y']].values
    n = len(home)
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier = np.argmin(dists) 
    p1 = home[ball_carrier]

    penalty = 0.0
    valid_passes = 0

    for j in range(n):
        if j == ball_carrier: continue
        p2 = home[j]
        dist = np.linalg.norm(p1 - p2)

        if dist > config.PASS_MAX_LEN:
            penalty += config.PASS_W_LONG * (dist - config.PASS_MAX_LEN)**2

        penalty += config.PASS_W_ANGLE * (1 - angle_score(p1, p2))

        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            penalty += config.PASS_W_BLOCK
        else:
            valid_passes += 1

    if valid_passes == 0:
        penalty += config.PASS_PENALTY_NO_OPTS

    return penalty

def cost_ball_support(df, ball_pos):
    dists = np.linalg.norm(df[['x', 'y']].values - ball_pos, axis=1)
    return np.min(dists) * config.BALL_SUPPORT_W_MULT