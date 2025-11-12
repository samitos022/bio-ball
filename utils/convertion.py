import numpy as np
import pandas as pd

def dict_to_array(position_dict, player_order=None):
    phases = ["Possesso offensivo", "Possesso difensivo", "Fase difensiva"]
    vector = []

    for phase in phases:
        df = position_dict.get(phase)
        if df is None:
            continue
        
        if player_order is not None:
            df = df.loc[player_order]

        for _, row in df.iterrows():
            vector.extend([row["x"], row["y"]])

    return np.array(vector)

def array_to_dict(vector, player_order, phases=["Possesso offensivo", "Possesso difensivo", "Fase difensiva"]):
    positions = {}
    n_players = len(player_order)
    data = vector.reshape(len(phases), n_players, 2)

    for i, phase in enumerate(phases):
        df = pd.DataFrame(data[i], columns=["x", "y"], index=player_order)
        positions[phase] = df

    return positions
