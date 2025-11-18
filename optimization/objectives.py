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

def calculate_pitch_control_objective(home_coords, away_coords, ball_pos):
    """
    Calcola il valore aggregato di Pitch Control per la squadra di casa.
    Restituisce un valore da minimizzare (quindi -PitchControl).
    """
    