import numpy as np
import config 

def penalty_total(positions):
    """
    Calcola le penalità strutturali (fuori campo, collisioni, ordine, transizione).
    Usa i pesi definiti in config.py.
    """
    penalty = 0.0
    phases = list(positions.keys()) 

    # Riferimento (fase iniziale)
    df_ref = positions[phases[0]]
    ref_x = df_ref["x"].values
    ref_y = df_ref["y"].values

    for phase_name, df in positions.items():
        pos = df[["x", "y"]].to_numpy()
        current_x = pos[:, 0]
        current_y = pos[:, 1]
        n_players = len(pos)

        # 1. BOUNDARIES (Fuori campo)
        out_x = (current_x < 0) | (current_x > config.FIELD_LIMITS[0])
        out_y = (current_y < 0) | (current_y > config.FIELD_LIMITS[1])
        penalty += config.PENALTY_W_BOUNDARY * (np.sum(out_x) + np.sum(out_y))
        
        # 2. PROXIMITY & ORDER
        for i in range(n_players):
            for j in range(i + 1, n_players):
                # A. Collisioni
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < config.MIN_DIST_PLAYER:
                    penalty += config.PENALTY_W_PROXIMITY * (config.MIN_DIST_PLAYER - dist) ** 2

                # B. Ordine Relativo (solo per frame successivi al primo)
                if phase_name != phases[0]:
                    # Penalità se i giocatori si incrociano in modo innaturale
                    if ref_x[i] < ref_x[j] and current_x[i] > current_x[j]:
                         penalty += config.PENALTY_W_ORDER * (current_x[i] - current_x[j])
                    
                    if ref_y[i] < ref_y[j] - 0.1 and current_y[i] > current_y[j]:
                         penalty += config.PENALTY_W_ORDER * (current_y[i] - current_y[j])

    # 3. TRANSITION (Spostamento eccessivo tra fasi)
    for i in range(len(phases) - 1):
        df1 = positions[phases[i]]
        df2 = positions[phases[i + 1]]

        # Assicuriamoci che l'ordine dei giocatori sia allineato tramite indice
        common = df1.index.intersection(df2.index)
        p1 = df1.loc[common, ["x", "y"]].to_numpy()
        p2 = df2.loc[common, ["x", "y"]].to_numpy()

        diffs = np.linalg.norm(p2 - p1, axis=1)
        penalty += config.PENALTY_W_TRANSITION * np.sum(diffs ** 2)

    return penalty