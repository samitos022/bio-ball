import numpy as np
import pandas as pd

def react_away_to_home(
    home_df,
    base_away_df,
    ball_pos,
    shift_horizontal=0.40,     # Aumentato → scalatura laterale molto più forte
    shift_vertical=0.30,       # Aumentato → squadra sale/aggressiva
    marking_strength=0.6,     # Aumentato → marcatura stretta
    pressing_force=0.1,       # NUOVO → va direttamente verso la palla
    role_constraints=True
):

    # Copia per sicurezza
    away = base_away_df.copy()

    # ===============================
    # 1) SCALATA ORIZZONTALE
    # ===============================
    shift_y = (ball_pos[1] - 0.5) * shift_horizontal
    away["y"] += shift_y

    # ===============================
    # 2) SCALATA VERTICALE
    # ===============================
    dist_ball_x = ball_pos[0] - 0.5
    shift_x = dist_ball_x * shift_vertical
    away["x"] -= shift_x  

    # ===============================
    # 3) REPARTI
    # ===============================
    n = len(away)

    if role_constraints and n >= 10:
        defenders_idx = range(0, 4)
        midfield_idx = range(4, 8)
        forwards_idx = range(8, 11)

        # Difensori → aggressività media (non vogliamo suicidi difensivi)
        away.iloc[defenders_idx, away.columns.get_loc("x")] += shift_x * 0.6
        away.iloc[defenders_idx, away.columns.get_loc("y")] += shift_y * 0.7

        # Centrocampo → molto aggressivi
        away.iloc[midfield_idx, away.columns.get_loc("x")] += shift_x * 1.4
        away.iloc[midfield_idx, away.columns.get_loc("y")] += shift_y * 1.6

        # Attaccanti → ultra-aggressivi (pressano in avanti)
        away.iloc[forwards_idx, away.columns.get_loc("x")] += shift_x * 1.8
        away.iloc[forwards_idx, away.columns.get_loc("y")] += shift_y * 2.0

    # ===============================
    # 4) MARCATURA INDIVIDUALE
    # ===============================
    home_positions = home_df[["x","y"]].values
    away_positions = away[["x","y"]].values

    for i in range(len(away_positions)):
        dists = np.linalg.norm(home_positions - away_positions[i], axis=1)
        closest_idx = np.argmin(dists)

        target = home_positions[closest_idx]
        current = away_positions[i]
        vec = target - current

        # marcatura più aggressiva
        away.iloc[i, away.columns.get_loc("x")] += marking_strength * vec[0]
        away.iloc[i, away.columns.get_loc("y")] += marking_strength * vec[1]

    # ===============================
    # 5) PRESSING DIRETTO SULLA PALLA
    # ===============================
    for i in range(len(away_positions)):
        vec_to_ball = np.array(ball_pos) - away_positions[i]
        away.iloc[i, away.columns.get_loc("x")] += pressing_force * vec_to_ball[0]
        away.iloc[i, away.columns.get_loc("y")] += pressing_force * vec_to_ball[1]

    # ===============================
    # 6) CLIP DEI VALORI
    # ===============================
    away["x"] = away["x"].clip(0.0, 1.0)
    away["y"] = away["y"].clip(0.0, 1.0)

    return away