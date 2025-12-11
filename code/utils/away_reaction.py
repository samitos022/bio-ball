import numpy as np
import pandas as pd  

def react_away_to_home(
    home_df,
    base_away_df,
    ball_pos,
    keeper_factor=0.03,
    block_lateral=0.30,     # molto più forte → squadra slitta verso lato palla
    block_vertical=0.20,    # sale/scende molto di più → più realistica
    shape_compactness=0.35, # compattezza reparti
    marking_factor=0.12,    # marcatura più evidente
    max_marking_distance=0.25,
    pressing_players=3,     # aumenta giocatori che reagiscono
    pressing_factor=0.12    # pressing più "vero"
):
    away = base_away_df.copy().reset_index(drop=True)

    n = len(away)
    keeper_idx = 0
    field_idx = np.arange(1, n)

    away_xy = away[["x", "y"]].to_numpy(float)
    home_xy = home_df[["x", "y"]].to_numpy(float)
    ball_pos = np.asarray(ball_pos, float)

    # ================
    # 1) SCALATA DEL BLOCCO
    # ================
    mean_x = away_xy[field_idx, 0].mean()
    mean_y = away_xy[field_idx, 1].mean()

    # LATERALE verso il lato palla
    away_xy[field_idx, 1] += (ball_pos[1] - mean_y) * block_lateral

    # VERTICALE verso la profondità palla
    away_xy[field_idx, 0] += (ball_pos[0] - mean_x) * block_vertical

    # ================
    # 2) ASSEGNAZIONE DEI REPARTI
    # ================
    ordered = field_idx[np.argsort(away_xy[field_idx, 0])]

    if len(ordered) >= 10:
        DEF = ordered[0:4]
        MID = ordered[4:7]
        ATT = ordered[7:10]
    else:
        # fallback semplice
        k = len(ordered)//3
        DEF = ordered[:k]
        MID = ordered[k:2*k]
        ATT = ordered[2*k:]

    # DIFESA: compatta
    away_xy[DEF] += (away_xy[DEF].mean(axis=0) - away_xy[DEF]) * shape_compactness

    # CENTROCAMPO: più dinamico
    away_xy[MID] += (away_xy[MID].mean(axis=0) - away_xy[MID]) * (shape_compactness * 0.6)

    # ATTACCO: reagisce molto alla palla
    away_xy[ATT] += (ball_pos - away_xy[ATT]) * 0.25


    # ================
    # 3) MARCATURA UOMO (MODERATA)
    # ================
    for i in field_idx:
        dists = np.linalg.norm(home_xy - away_xy[i], axis=1)
        closest = np.argmin(dists)

        vec = home_xy[closest] - away_xy[i]
        d = np.linalg.norm(vec)

        if d < max_marking_distance:
            away_xy[i] += marking_factor * vec


    # ================
    # 4) PRESSING (3 giocatori più vicini)
    # ================
    ball_dists = np.linalg.norm(away_xy[field_idx] - ball_pos, axis=1)
    press = field_idx[np.argsort(ball_dists)[:pressing_players]]

    for i in press:
        away_xy[i] += pressing_factor * (ball_pos - away_xy[i])


    # ================
    # 5) PORTIERE (quasi fermo)
    # ================
    gk = away_xy[keeper_idx]
    if ball_pos[0] > 0.80:
        away_xy[keeper_idx] += keeper_factor * (ball_pos - gk)


    # ================
    # 6) CLIP CAMPO
    # ================
    away_xy = np.clip(away_xy, 0, 1)
    away["x"], away["y"] = away_xy[:, 0], away_xy[:, 1]

    return away
