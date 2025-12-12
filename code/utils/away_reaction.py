import numpy as np
import pandas as pd

def react_away_to_home(
    home_df,
    base_away_df,
    ball_pos,
    keeper_factor=0.03,
    block_lateral=0.30,     # Shifts team towards the ball side
    block_vertical=0.20,    # Adjusts defensive line height
    shape_compactness=0.35, # Tightens distances within tactical lines
    marking_factor=0.12,    # Intensity of man-marking
    max_marking_distance=0.25,
    pressing_players=3,     # Number of players chasing the ball
    pressing_factor=0.12    # Intensity of pressing
):
    """
    Calculates the reactive positions of the Away team based on the ball
    and Home team positions using a deterministic heuristic model.
    """
    # Create a copy to avoid modifying the original data
    away = base_away_df.copy().reset_index(drop=True)

    n = len(away)
    keeper_idx = 0
    outfield_idx = np.arange(1, n) # Assumes index 0 is always the GK

    # Convert to numpy for vectorization
    away_xy = away[["x", "y"]].to_numpy(float)
    home_xy = home_df[["x", "y"]].to_numpy(float)
    ball_pos = np.asarray(ball_pos, float)

    # 1. BLOCK SHIFTING (Team Geometry)
    mean_x = away_xy[outfield_idx, 0].mean()
    mean_y = away_xy[outfield_idx, 1].mean()

    # Lateral shift towards the ball side
    away_xy[outfield_idx, 1] += (ball_pos[1] - mean_y) * block_lateral

    # Vertical shift (depth) based on ball position
    away_xy[outfield_idx, 0] += (ball_pos[0] - mean_x) * block_vertical

    # 2. TACTICAL LINES ADJUSTMENT
    # Sort outfield players by X to identify Defenders, Midfielders, Attackers
    sorted_indices = outfield_idx[np.argsort(away_xy[outfield_idx, 0])]

    if len(sorted_indices) >= 10:
        def_idx = sorted_indices[0:4]
        mid_idx = sorted_indices[4:7]
        att_idx = sorted_indices[7:10]
    else:
        # Fallback: split into thirds
        k = len(sorted_indices) // 3
        def_idx = sorted_indices[:k]
        mid_idx = sorted_indices[k:2*k]
        att_idx = sorted_indices[2*k:]

    # Defense: High compactness
    away_xy[def_idx] += (away_xy[def_idx].mean(axis=0) - away_xy[def_idx]) * shape_compactness

    # Midfield: Moderate compactness
    away_xy[mid_idx] += (away_xy[mid_idx].mean(axis=0) - away_xy[mid_idx]) * (shape_compactness * 0.6)

    # Attack: Reacts aggressively to ball position
    away_xy[att_idx] += (ball_pos - away_xy[att_idx]) * 0.25

    # 3. MAN MARKING (Proximity based)
    for i in outfield_idx:
        # Find nearest opponent
        dists = np.linalg.norm(home_xy - away_xy[i], axis=1)
        closest_home_idx = np.argmin(dists)

        vec = home_xy[closest_home_idx] - away_xy[i]
        dist = np.linalg.norm(vec)

        # Move closer if within marking range
        if dist < max_marking_distance:
            away_xy[i] += marking_factor * vec

    # 4. PRESSING (Ball Chasing)
    # Identify the N players closest to the ball
    dist_to_ball = np.linalg.norm(away_xy[outfield_idx] - ball_pos, axis=1)
    pressing_indices = outfield_idx[np.argsort(dist_to_ball)[:pressing_players]]

    for i in pressing_indices:
        away_xy[i] += pressing_factor * (ball_pos - away_xy[i])

    # 5. GOALKEEPER MOVEMENT
    gk_pos = away_xy[keeper_idx]
    # GK adjusts slightly if ball is deep in their half
    if ball_pos[0] > 0.80:
        away_xy[keeper_idx] += keeper_factor * (ball_pos - gk_pos)

    # 6. BOUNDARY CLAMPING
    away_xy = np.clip(away_xy, 0, 1)
    
    # Update DataFrame
    away["x"], away["y"] = away_xy[:, 0], away_xy[:, 1]

    return away