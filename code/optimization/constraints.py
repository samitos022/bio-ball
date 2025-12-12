import numpy as np
import config 

def penalty_total(positions, detailed=False):

    penalty = 0.0
    phases = list(positions.keys()) 

    df_ref = positions[phases[0]]
    ref_x = df_ref["x"].values
    ref_y = df_ref["y"].values

    pen_boundary = 0.0
    pen_collision = 0.0
    pen_order = 0.0
    pen_transition = 0.0
    pen_goalkeeper = 0.0

    for phase_name, df in positions.items():

        pos = df[["x", "y"]].to_numpy()
        current_x = pos[:, 0]
        current_y = pos[:, 1]
        n_players = len(pos)

        # 1. BOUNDARIES
        out_x = (current_x < 0) | (current_x > config.FIELD_LIMITS[0])
        out_y = (current_y < 0) | (current_y > config.FIELD_LIMITS[1])
        pen_boundary += config.PENALTY_W_BOUNDARY * (np.sum(out_x) + np.sum(out_y))

        # GOALKEEPER RESTRICTION
        gk_x = current_x[0]
        gk_y = current_y[0]

        if not (config.GOALKEEPER_AREA["x_min"] <= gk_x <= config.GOALKEEPER_AREA["x_max"] and
                config.GOALKEEPER_AREA["y_min"] <= gk_y <= config.GOALKEEPER_AREA["y_max"]):

            dx = max(0, config.GOALKEEPER_AREA["x_min"] - gk_x, gk_x - config.GOALKEEPER_AREA["x_max"])
            dy = max(0, config.GOALKEEPER_AREA["y_min"] - gk_y, gk_y - config.GOALKEEPER_AREA["y_max"])
            dist_out = np.sqrt(dx*dx + dy*dy)

            pen_goalkeeper += config.PENALTY_W_GOALKEEPER * (dist_out ** 2)


        # 2. COLLISIONS & ORDER
        alpha = 40 # Steepness coefficient; adjust for more/less severity

        for i in range(n_players):
            for j in range(i + 1, n_players):

                dist = np.linalg.norm(pos[i] - pos[j])

                if dist < config.MIN_DIST_PLAYER:
                    delta = config.MIN_DIST_PLAYER - dist

                    exp_pen = config.PENALTY_W_PROXIMITY * np.exp(alpha * delta)

                    if dist < config.MIN_DIST_PLAYER * 0.5:
                        exp_pen *= 3  

                    pen_collision += exp_pen

    # 3. TRANSITION SMOOTHNESS
    for i in range(len(phases) - 1):
        df1 = positions[phases[i]]
        df2 = positions[phases[i + 1]]
        common = df1.index.intersection(df2.index)
        diffs = np.linalg.norm(
            df2.loc[common, ["x", "y"]].to_numpy() -
            df1.loc[common, ["x", "y"]].to_numpy(),
            axis=1
        )
        pen_transition += config.PENALTY_W_TRANSITION * np.sum(diffs ** 2)

    total_penalty = (
        pen_boundary +
        pen_collision +
        pen_order +
        pen_transition +
        pen_goalkeeper
    )

    if detailed:
        return {
            "total": total_penalty,
            "boundary": pen_boundary,
            "collision": pen_collision,
            "order": pen_order,
            "transition": pen_transition,
            "goalkeeper": pen_goalkeeper
        }

    return total_penalty
