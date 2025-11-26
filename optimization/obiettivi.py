import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from utils.conversion import flat_to_formation

def cost_boundaries(vector):
    """Penalità 'Muro' se i giocatori escono dal campo (0-1)."""
    # Calcola quanto si sfora sotto 0 e sopra 1
    out_of_bounds = np.sum(np.maximum(0, -vector)**2 + np.maximum(0, vector - 1)**2)
    return out_of_bounds * 100000.0 

def cost_coverage(df):
    """Massimizzare l'area (Convex Hull). Ritorna (1 - Area)."""
    points = df[['x', 'y']].values
    if len(points) < 3: return 1.0
    
    try:
        hull = ConvexHull(points)
        area = hull.volume # In 2D 'volume' è l'area
        return 1.0 - area 
    except:
        return 1.0

def cost_passing_lanes(df_home, obstacles_array, ball_pos):
    """Calcola penalità per linee di passaggio bloccate e posizionamento."""
    home_pos = df_home[['x', 'y']].values
    n_players = len(home_pos)
    total_penalty = 0.0
    
    for i in range(n_players):
        p1 = home_pos[i]
        
        # 1. Supporto alla palla: penalità se troppo lontani
        dist_ball = np.linalg.norm(p1 - ball_pos)
        total_penalty += dist_ball * 0.2

        for j in range(i + 1, n_players):
            p2 = home_pos[j]
            
            # 2. Distanza reciproca (Spacing)
            dist = np.linalg.norm(p1 - p2)
            if dist < 0.05: total_penalty += 5.0 # Troppo vicini (5m)
            if dist > 0.45: total_penalty += 1.0 # Troppo lontani (45m)
            
            # 3. Linee di passaggio vs Ostacoli
            vec_pass = p2 - p1
            len_pass = np.linalg.norm(vec_pass)
            if len_pass == 0: continue
            
            # Controllo blocco ostacoli
            blocked = False
            for obs in obstacles_array:
                # Proiezione punto su segmento
                t = np.dot(obs - p1, vec_pass) / (len_pass**2)
                t = np.clip(t, 0, 1)
                projection = p1 + t * vec_pass
                dist_obs = np.linalg.norm(obs - projection)
                
                # Se l'ostacolo è a meno di 3m dalla linea di passaggio
                if dist_obs < 0.03: 
                    blocked = True
                    break
            
            if blocked:
                total_penalty += 2.5 # Penalità linea intercettata

    return total_penalty

# --- FUNZIONE PRINCIPALE (WRAPPER) ---

def objective_function(vector, args):
    """Funzione target per CMA-ES."""
    # Unpack degli argomenti passati da run_optimization
    player_names, obstacles_array, ball_pos, initial_df_ref = args
    
    # Decoding
    df_candidate = flat_to_formation(vector, player_names)
    
    # Verifica immediata boundaries per risparmiare tempo
    c_bounds = cost_boundaries(vector)
    if c_bounds > 1000: return c_bounds
        
    # Calcolo costi
    c_pass = cost_passing_lanes(df_candidate, obstacles_array, ball_pos)
    c_cover = cost_coverage(df_candidate)
    
    # Pesi (Tuning)
    w_pass = 2.0
    w_cover = 10.0
    
    total_cost = c_bounds + (c_pass * w_pass) + (c_cover * w_cover)
                 
    return total_cost