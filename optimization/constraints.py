import numpy as np

def penalty_total(positions, field_limits=(1.0, 1.0), 
                  min_dist=0.08,
                  transition_weight=1.0, 
                  boundary_weight=1000.0,
                  proximity_weight=500.0,
                  order_weight=10.0):
    
    penalty = 0.0
    phases = list(positions.keys()) # ["Start", "Candidate"]

    df_ref = positions[phases[0]]
    ref_x = df_ref["x"].values
    ref_y = df_ref["y"].values

    for phase_name, df in positions.items():
        pos = df[["x", "y"]].to_numpy()
        current_x = pos[:, 0]
        current_y = pos[:, 1]

        # 1. BOUNDARIES 
        out_of_bounds_x = np.sum((current_x < 0) | (current_x > field_limits[0]))
        out_of_bounds_y = np.sum((current_y < 0) | (current_y > field_limits[1]))
        penalty += boundary_weight * (out_of_bounds_x + out_of_bounds_y)

        n_players = len(pos)
        
        # 2. PROXIMITY & ORDER (Loop sui giocatori)
        for i in range(n_players):
            for j in range(i + 1, n_players):
                
                # A. Collisioni
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_dist:
                    penalty += proximity_weight * (min_dist - dist) ** 2

                # B. Mantenimento Ordine Relativo
                # Solo per la fase candidata
                if phase_name != phases[0]:
                    
                    # Controllo Longitudinale (Difensori restano dietro Attaccanti)
                    if ref_x[i] < ref_x[j] and current_x[i] > current_x[j]:
                         penalty += order_weight * (current_x[i] - current_x[j])
                    
                    # Controllo Laterale (Sinistra resta a Sinistra)
                    if ref_y[i] < ref_y[j] - 0.1 and current_y[i] > current_y[j]:
                         penalty += order_weight * (current_y[i] - current_y[j])

    # 3. TRANSITION (Ancoraggio alla posizione precedente)
    for i in range(len(phases) - 1):
        df1 = positions[phases[i]]
        df2 = positions[phases[i + 1]]

        common_players = df1.index.intersection(df2.index)
        p1 = df1.loc[common_players, ["x", "y"]].to_numpy()
        p2 = df2.loc[common_players, ["x", "y"]].to_numpy()

        diffs = np.linalg.norm(p2 - p1, axis=1)
        penalty += transition_weight * np.sum(diffs ** 2)

    return penalty
