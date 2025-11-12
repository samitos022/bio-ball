import numpy as np

def penalty_total(positions, field_limits=(1.0, 1.0), min_dist=0.05, transition_weight=1.0, boundary_weight=5.0, proximity_weight=2.0):
    penalty = 0.0
    phases = list(positions.keys())

    for phase_name, df in positions.items():
        pos = df[["x", "y"]].to_numpy()

        out_of_bounds_x = np.sum((pos[:, 0] < 0) | (pos[:, 0] > field_limits[0]))
        out_of_bounds_y = np.sum((pos[:, 1] < 0) | (pos[:, 1] > field_limits[1]))
        penalty += boundary_weight * (out_of_bounds_x + out_of_bounds_y)

        n_players = len(pos)
        
        for i in range(n_players):
            for j in range(i + 1, n_players):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_dist:
                    penalty += proximity_weight * (min_dist - dist) ** 2

    for i in range(len(phases) - 1):
        df1 = positions[phases[i]]
        df2 = positions[phases[i + 1]]

        common_players = df1.index.intersection(df2.index)
        p1 = df1.loc[common_players, ["x", "y"]].to_numpy()
        p2 = df2.loc[common_players, ["x", "y"]].to_numpy()

        diffs = np.linalg.norm(p2 - p1, axis=1)
        penalty += transition_weight * np.sum(diffs ** 2)

    return penalty
