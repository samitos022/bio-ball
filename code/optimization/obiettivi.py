import numpy as np
from scipy.spatial import ConvexHull
from constraints import penalty_total

def coverage_field(players):
    points = players[["x", "y"]].to_numpy() 

    if len(points) < 3:
        return 0

    area = ConvexHull(points).volume

    return area

def coverage_field_penalty(positions, w1=1, w2=1, w3=1):
    coverage_offensive = coverage_field(positions["Possesso offensivo"])
    coverage_difensive = coverage_field(positions["Possesso difensivo"])
    coverage_pure_difensive = coverage_field(positions["Fase difensiva"])

    penalty = w1 * (1 - coverage_offensive) + w2 * (1 - coverage_difensive) + w3 * (1 - coverage_pure_difensive)

    return penalty

def transition_cost(positions, w=1):
    phases = list(positions.keys())
    total = 0

    for i in range(len(phases) - 1):
        for j in range(i+1, len(phases)):
            p1 = positions[phases[i]][["x","y"]].to_numpy()
            p2 = positions[phases[j]][["x","y"]].to_numpy()

            diff = np.linalg.norm(p2 - p1, axis=1)
            total += np.sum(diff ** 2)

    return w * total

def point_line_distance(point, a, b):
    a = np.array(a)
    b = np.array(b)
    p = np.array(point)

    if np.all(a == b):
        return np.linalg.norm(p - a)

    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    t = np.clip(t, 0, 1)
    projection = a + t * (b - a)
    return np.linalg.norm(p - projection)

def angle_score(p_i, p_j, team_direction=np.array([1.0, 0.0])):
    v = p_j - p_i
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return 0.0
        
    v = v / norm_v
    cosang = np.dot(v, team_direction)

    return (cosang + 1) / 2

def passing_lanes_penalty(positions_home, positions_away, block_threshold=0.05, weight_block=5.0, weight_distance=0.5, weight_angle=0.5):
    home = positions_home[["x", "y"]].to_numpy()
    away = positions_away[["x", "y"]].to_numpy()

    n = len(home)
    penalty = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            p_i = home[i]
            p_j = home[j]

            dist = np.linalg.norm(p_j - p_i)
            dist_penalty = weight_distance * dist

            ang_bonus = angle_score(p_i, p_j)
            ang_penalty = weight_angle * (1 - ang_bonus)

            blocked = False
            for opp in away:
                d = point_line_distance(opp, p_i, p_j)
                if d < block_threshold:
                    blocked = True
                    break

            block_penalty = weight_block if blocked else 0.0

            penalty += dist_penalty + ang_penalty + block_penalty

    return penalty

def evaluate(positions_home, positions_away, w_coverage=1.0, w_transition=1.0, w_passing_off=1.0, w_passing_def=0.7, w_passing_pure_def=0.4):
    coverage_pen = coverage_field_penalty(positions_home)

    transition_pen = transition_cost(positions_home)

    passing_off = passing_lanes_penalty(
        positions_home["Possesso offensivo"],
        positions_away["Possesso offensivo"],
    )

    passing_def = passing_lanes_penalty(
        positions_home["Possesso difensivo"],
        positions_away["Possesso difensivo"],
    )

    passing_pure_def = passing_lanes_penalty(
        positions_home["Fase difensiva"],
        positions_away["Fase difensiva"],
    )

    total_passing_pen = (w_passing_off * passing_off + w_passing_def * passing_def + w_passing_pure_def * passing_pure_def)
    
    constraints_pen = penalty_total(positions_home)
    
    penalty = w_coverage * coverage_pen + w_transition * transition_pen + total_passing_pen + constraints_pen

    return penalty