import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import config

# --- HELPER FUNCTIONS ---
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

def exclude_goalkeeper(df_home):
    """Removes the goalkeeper (identified as the player with min X)."""
    if isinstance(df_home, pd.DataFrame):
        gk_idx = df_home['x'].idxmin()
        return df_home.drop(index=gk_idx)
    min_x_idx = np.argmin(df_home[:, 0])
    return np.delete(df_home, min_x_idx, axis=0)

# --- OFFENSIVE OBJECTIVES ---
def cost_coverage(df, detailed=False):
    """
    Calculates tactical field coverage.
    - Central/Advanced zones: Max value
    - Flanks: Medium value
    - Defensive corners: Min value
    """
    df_outfield = exclude_goalkeeper(df)
    points = df_outfield[['x', 'y']].values
    
    if len(points) < 3:
        return {"total": 100.0, "coverage": 0.0} if detailed else 100.0
    
    grid_size = 20
    player_radius = 0.12
    
    # Create Grid
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size)
    )
    cells = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    
    # Calculate Tactical Weights
    x, y = cells[:, 0], cells[:, 1]
    
    # 1. Advancement: Linear increase (0.5 -> 1.5)
    advancement = 0.5 + x
    
    # 2. Centrality: Gaussian (Max at center y=0.5)
    centrality = np.exp(-5 * (y - 0.5)**2)
    
    # 3. Defensive corners penalty
    corner_penalty = np.where((x < 0.3) & ((y < 0.2) | (y > 0.8)), 0.3, 1.0)
    
    weights = advancement * centrality * corner_penalty
    
    # Check Coverage
    covered = np.zeros(len(cells), dtype=bool)
    for point in points:
        distances = np.linalg.norm(cells - point, axis=1)
        covered |= (distances <= player_radius)
    
    # Metrics
    weighted_coverage = np.sum(weights[covered])
    max_coverage = np.sum(weights)
    coverage_ratio = weighted_coverage / max_coverage
    
    cost = (1.0 - coverage_ratio) * 3
    
    if detailed:
        return {
            "total": cost,
            "coverage_ratio": np.sum(covered) / len(cells),
            "weighted_ratio": coverage_ratio,
            "avg_x": np.mean(points[:, 0])
        }
    
    return cost

def cost_offside_avoidance(df_home, obstacles_array, ball_pos, attacking_dir='right', detailed=False):
    """(Attacking Possession) Penalizes players who are offside."""
    home_x = df_home['x'].values
    opp_x = obstacles_array[:, 0]
    offside_dist = 0.0
    
    if len(opp_x) >= 2:
        if attacking_dir == 'right':
            line = np.sort(opp_x)[-2]
            eff_line = max(line, ball_pos[0])
            mask = home_x > eff_line
            if np.any(mask): offside_dist = np.sum(home_x[mask] - eff_line)
        else:
            line = np.sort(opp_x)[1]
            eff_line = min(line, ball_pos[0])
            mask = home_x < eff_line
            if np.any(mask): offside_dist = np.sum(eff_line - home_x[mask])

    if detailed: return {"total": offside_dist, "meters": offside_dist}
    return offside_dist

# --- DEFENSIVE OBJECTIVES ---

def cost_marking(df_home, obstacles_array, detailed=False):
    """
    (Defensive Phase) Ensures every outfield opponent is marked.
    Excludes the opponent GK to avoid useless pressing.
    """
    home = df_home[['x', 'y']].values
    opps = obstacles_array
    
    # Assume opponent GK is the player with max X (furthest from our goal at 0)
    if len(opps) > 0:
        gk_idx = np.argmax(opps[:, 0])
        targets = np.delete(opps, gk_idx, axis=0)
    else:
        targets = opps
        
    sum_sq_dist = 0.0
    
    if len(targets) == 0:
        if detailed: return {"total": 0.0, "avg_dist": 0.0}
        return 0.0
    
    # Find nearest defender for each opponent
    for opp in targets:
        dists = np.linalg.norm(home - opp, axis=1)
        min_d = np.min(dists)
        # Quadratic penalty to severely punish unmarked opponents
        sum_sq_dist += min_d ** 2

    # Normalize
    cost = sum_sq_dist / len(targets)
    cost_scaled = cost * 10.0
    
    if detailed:
        rmse = np.sqrt(sum_sq_dist / len(targets))
        return {"total": cost_scaled, "avg_dist": rmse}
    
    return cost_scaled

def cost_defensive_compactness(df_home, detailed=False):
    """(Defensive Phase) Minimizes dispersion from the team centroid (Excluding GK)."""
    df_outfield = exclude_goalkeeper(df_home)
    coords = df_outfield[['x', 'y']].values
    
    centroid = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    dispersion = np.mean(dists)
    
    final_cost = dispersion * 5.0
    
    if detailed:
        return {"total": final_cost, "dispersion": dispersion}
    return final_cost

def cost_defensive_line_height(df_home, ball_pos, detailed=False):
    """
    (Defensive Phase) Dynamic and Elastic Line Height.
    - Calculates the line based on the mean of the last defender(s).
    - Ideal distance varies: Tight if ball is near goal, Spaced if ball is far.
    """
    outfield_x = exclude_goalkeeper(df_home)['x'].values
    
    if len(outfield_x) < 3: 
        line_x = np.min(outfield_x) if len(outfield_x) > 0 else 0.0
    else:
        sorted_x = np.sort(outfield_x)
        # Calculates mean of the last defender(s)
        line_x = np.mean(sorted_x[:1])

    ball_x = ball_pos[0]

    # Calculate Dynamic Ideal Distance (Elasticity)
    # Formula: 5m base + (25% of ball position)
    dynamic_buffer = 0.05 + (ball_x * 0.25)
    
    target_x = ball_x - dynamic_buffer
    
    # Field constraints
    target_x = max(target_x, 0.05)
    target_x = min(target_x, 0.60) 

    cost = 0.0
    
    # Penalty: Too Deep
    if line_x < target_x:
        cost += (target_x - line_x) * 8.0 
        
    # Penalty: Over Ball (Panic)
    if line_x > ball_x:
        cost += (line_x - ball_x) * 50.0

    if detailed:
        return {
            "total": cost,
            "line_x": line_x,
            "target_dynamic": target_x,
            "ideal_dist": dynamic_buffer,
            "ball_x": ball_x
        }
        
    return cost

def cost_preventive_marking(df_home, obstacles_array, detailed=False):
    """(Attacking Possession) Preventive Marking against counterattacks."""
    home = df_home[['x', 'y']].values
    # Threats are opponents in our half
    threats = [opp for opp in obstacles_array if opp[0] < 0.5]
    
    if not threats: 
        if detailed: return {"total": 0.0, "threats": 0}
        return 0.0
    
    cost = 0.0
    for threat in threats:
        dists = np.linalg.norm(home - threat, axis=1)
        min_d = np.min(dists)
        cost += min_d
        
    final_cost = cost * 5.0
    
    if detailed:
        return {"total": final_cost, "threats": len(threats), "raw_dist": cost}
    return final_cost

# --- COMMON OBJECTIVES ---

def cost_ball_pressure(df_home, ball_pos, detailed=False):
    """
    (Defensive/Offensive) Distance of the closest player to the ball.
    Defensive = Pressing, Offensive = Support.
    """
    dists = np.linalg.norm(df_home[['x', 'y']].values - ball_pos, axis=1)
    min_dist = np.min(dists)
    
    cost = min_dist * 20.0 
    
    if detailed: return {"total": cost, "dist": min_dist}
    return cost

def cost_passing_lanes(df_home, obstacles_array, ball_pos, phase_type="Attacking possession", detailed=False):
    """
    Calculates passing availability cost using a HYBRID LOGIC (Saturation + Bonus).
    
    1. Saturation (Exp): Punishes severely if below 'Target Score' (Ensures min options).
    2. Bonus (Linear): Rewards extra quality (Encourages excellence).
    """
    home = df_home[['x', 'y']].values
    player_names = df_home.index.tolist()
    n = len(home)
    
    # Identify Carrier
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier_idx = np.argmin(dists) 
    p1 = home[ball_carrier_idx]
    ball_carrier_name = player_names[ball_carrier_idx]

    total_quality_score = 0.0
    valid_options_count = 0
    blocked_count = 0
    
    valid_receivers = []
    
    is_offensive = "offensivo" in phase_type.lower()
    target_score = config.PASS_TARGET_SCORE_OFF if is_offensive else config.PASS_TARGET_SCORE_DEF

    for j in range(n):
        if j == ball_carrier_idx: continue
        p2 = home[j]
        receiver_name = player_names[j]
        
        # 1. Check Block
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            blocked_count += 1
            continue
            
        # 2. Check Length/Proximity
        dist = np.linalg.norm(p1 - p2)
        
        if dist > config.PASS_MAX_LEN:
            continue
            
        # Filter crowding (< 5m is usually useless)
        if dist < 0.05: 
            continue

        # Calculate Score
        pass_val = 1.0
        
        if is_offensive:
            # Attacking Possession: Reward verticality
            ang = angle_score(p1, p2)
            angle_multiplier = 1.0 + (ang * 0.5) 
            pass_val *= angle_multiplier
            
        else:
            # Defensive Possession: Reward safety
            # Longer passes are riskier
            if dist > 0.20: 
                pass_val *= 0.8

        total_quality_score += pass_val
        valid_options_count += 1
        
        valid_receivers.append(f"{receiver_name} ({pass_val:.1f})")

    # --- 3. Calculate Final Hybrid Cost ---
    
    # A. Safety Component (Exponential)
    saturation_cost = config.PASS_PENALTY_NO_OPTS * np.exp(-total_quality_score / target_score)
    
    # B. Excellence Component (Linear Bonus)
    reward_factor = 0.5 
    linear_bonus = total_quality_score * reward_factor
    
    total_cost = saturation_cost - linear_bonus

    if detailed:
        return {
            "total": total_cost,
            "quality_score": total_quality_score,
            "target_score": target_score,
            "valid_count": valid_options_count,
            "blocked_count": blocked_count,
            "carrier": ball_carrier_name,
            "receivers": valid_receivers,
            "cost_sat": saturation_cost,
            "cost_bonus": -linear_bonus
        }
        
    return total_cost