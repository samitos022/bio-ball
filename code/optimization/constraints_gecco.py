import numpy as np
import config_gecco as cfg

def calculate_penalties(home_coords, away_coords, ball_pos, phase_name, attacking_dir="right", detailed=False):
    
    home_x, home_y = home_coords[:, 0], home_coords[:, 1]
    is_attacking_right = (attacking_dir == "right")
    
    # 1. BOUNDARIES
    out_x_left, out_x_right = np.maximum(0, -home_x), np.maximum(0, home_x - cfg.FIELD_LIMITS[0])
    out_y_bottom, out_y_top = np.maximum(0, -home_y), np.maximum(0, home_y - cfg.FIELD_LIMITS[1])
    bound_error = np.sum(out_x_left**2 + out_x_right**2 + out_y_bottom**2 + out_y_top**2)
    pen_boundary = bound_error * cfg.PENALTY_W_BOUNDARY

    # 2. GOALKEEPER AREA
    gk_x, gk_y = home_x[0], home_y[0]
    x_min, x_max = cfg.GOALKEEPER_AREA["x_min"], cfg.GOALKEEPER_AREA["x_max"]
    y_min, y_max = cfg.GOALKEEPER_AREA["y_min"], cfg.GOALKEEPER_AREA["y_max"]
    
    if "Attacking" in phase_name or phase_name == "pa":
        x_max += 0.15 
    if not is_attacking_right:
        x_min, x_max = 1.0 - x_max, 1.0 - x_min
        
    dx_gk = max(0, x_min - gk_x, gk_x - x_max)
    dy_gk = max(0, y_min - gk_y, gk_y - y_max)
    pen_gk = (dx_gk**2 + dy_gk**2) * cfg.PENALTY_W_GOALKEEPER

    # 3. ANTI-COLLISIONE
    diffs = home_coords[:, np.newaxis, :] - home_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, 100.0)
    collisions = np.maximum(0, cfg.MIN_DIST_PLAYER - dists)
    pen_collisions = np.sum(collisions**2) * cfg.PENALTY_W_PROXIMITY * 0.5

    # 4. FUORIGIOCO
    pen_offside = 0.0
    if len(away_coords) >= 2:
        away_x = away_coords[:, 0]
        if is_attacking_right:
            second_last_opp_x = np.partition(away_x, -2)[-2] 
            offside_line = max(ball_pos[0], second_last_opp_x, 0.5) 
            offside_errors = np.maximum(0, home_x[1:] - offside_line)
        else:
            second_last_opp_x = np.partition(away_x, 1)[1]
            offside_line = min(ball_pos[0], second_last_opp_x, 0.5)
            offside_errors = np.maximum(0, offside_line - home_x[1:])
        pen_offside = np.sum(offside_errors**2) * cfg.PENALTY_W_OFFSIDE

    # 5. AZIONE SULLA PALLA
    dists_to_ball = np.linalg.norm(home_coords[1:] - ball_pos, axis=1)
    min_dist_to_ball = np.min(dists_to_ball)
    
    if "Attacking" in phase_name or "possession" in phase_name:
        dist_error = max(0, min_dist_to_ball - 0.005)
        pen_ball = (dist_error**2) * cfg.PENALTY_W_BALL
    else:
        dist_error = max(0, min_dist_to_ball - 0.03)
        pen_ball = (dist_error**2) * cfg.PENALTY_W_BALL
        
    total_penalty = pen_boundary + pen_gk + pen_collisions + pen_offside + pen_ball

    if detailed:
        return {
            "Total": total_penalty,
            "Boundaries": pen_boundary,
            "Goalkeeper": pen_gk,
            "Collisions": pen_collisions,
            "Offside": pen_offside,
            "Ball_Action": pen_ball
        }
    return total_penalty