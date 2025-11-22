import numpy as np
from scipy.spatial import ConvexHull

def coverage_field(players):
    points = players[["x", "y"]].to_numpy() 

    if len(points) < 3:
        return 0

    area = ConvexHull(points).volume

    return area

def coverage_field_penalty(positions, w1=1, w2=1, w3=1):
    coverage_offensive = coverage_field(positions["Possesso offensivo"])
    coverage_difensive = coverage_field(positions["Possesso difensivo"])
    coverage_pure_difensive = coverage_field(positions["Fase difensiva"])

    penalty = w1 * (1 - coverage_offensive) + w2 * (1 - coverage_difensive) + w3 * (1 - coverage_pure_difensive)

    return penalty

def transition_cost(positions, w=1):
    phases = list(positions.keys())
    total = 0

    for i in range(len(phases) - 1):
        for j in range(i+1, len(phases)):
            p1 = positions[phases[i]][["x","y"]].to_numpy()
            p2 = positions[phases[j]][["x","y"]].to_numpy()

            diff = np.linalg.norm(p2 - p1, axis=1)
            total += np.sum(diff ** 2)

    return w * total

def calculate_pitch_control_objective(home_coords, away_coords, ball_pos):
    """
    Calcola il valore aggregato di Pitch Control per la squadra di casa.
    Restituisce un valore da minimizzare (quindi -PitchControl).
    """
    