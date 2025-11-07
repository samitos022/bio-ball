# optimization/objectives.py
import numpy as np
from scipy.spatial import Voronoi

def calculate_voronoi_objective(player_coords):
    """
    Calcola un obiettivo basato sui diagrammi di Voronoi.
    Minimizza la varianza delle aree per una copertura uniforme.
    """
 

def calculate_pitch_control_objective(home_coords, away_coords, ball_pos):
    """
    Calcola il valore aggregato di Pitch Control per la squadra di casa.
    Restituisce un valore da minimizzare (quindi -PitchControl).
    """
    