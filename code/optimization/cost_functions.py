import numpy as np
from scipy.spatial import ConvexHull
import config

# --- HELPER FUNCTIONS ---
def point_line_distance(point, a, b):
    a, b, p = np.array(a), np.array(b), np.array(point)
    if np.all(a == b): return np.linalg.norm(p - a)
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p - (a + t * (b - a)))

def angle_score(p_i, p_j, team_direction=np.array([1.0, 0.0])):
    v = p_j - p_i
    norm = np.linalg.norm(v)
    if norm == 0: return 0.0
    return (np.dot(v/norm, team_direction) + 1) / 2

# --- OFFENSIVE OBJECTIVES ---

def cost_coverage(df, detailed=False):
    """(Offensive) Maximizes Pitch Coverage (1 - Area)"""
    points = df[['x', 'y']].values
    area = 0.0
    if len(points) >= 3:
        try:
            area = ConvexHull(points).volume
        except:
            area = 0.0
    
    cost = 1.0 - area
    if detailed: return {"total": cost, "raw_area": area}
    return cost

def cost_passing_lanes(df_home, obstacles_array, ball_pos, detailed=False):
    """
    Nuova logica: Premia la DISPONIBILITÀ di passaggi (quantità e qualità), 
    NON penalizza il fatto che alcuni compagni lontani siano marcati.
    """
    home = df_home[['x', 'y']].values
    n = len(home)
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier_idx = np.argmin(dists) 
    p1 = home[ball_carrier_idx]

    # Parametri interni
    MIN_OPTIONS_NEEDED = 3 # Vogliamo almeno 3 scarichi sicuri
    
    valid_options_count = 0
    quality_score_sum = 0.0
    
    pen_block_debug = 0 # Solo per report
    
    for j in range(n):
        if j == ball_carrier_idx: continue
        p2 = home[j]
        
        # 1. Check Blocco
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            pen_block_debug += 1
            continue # Passaggio bloccato: lo ignoriamo, non lo penalizziamo!
            
        # 2. Se passa il blocco, valutiamo la qualità
        dist = np.linalg.norm(p1 - p2)
        
        # Ignoriamo passaggi troppo lunghi (non contano come opzioni valide)
        if dist > config.PASS_MAX_LEN:
            continue
            
        # Ignoriamo passaggi troppo corti (ammucchiata inutile)
        if dist < 0.05: # 5 metri
            continue

        # Calcolo Score Qualità (Angolo + Distanza ideale)
        ang_score = angle_score(p1, p2) # 1.0 se avanti, 0.0 se indietro
        
        # Un passaggio è un'opzione valida
        valid_options_count += 1
        quality_score_sum += ang_score

    # --- CALCOLO COSTO ---
    # L'obiettivo è minimizzare il costo.
    # Se abbiamo poche opzioni, costo alto.
    # Se abbiamo tante opzioni, costo basso (o addirittura negativo/bonus).
    
    missing_options = max(0, MIN_OPTIONS_NEEDED - valid_options_count)
    
    # Penalità enorme se non hai opzioni (es. 10.0 per ogni opzione mancante)
    cost_quantity = missing_options * config.PASS_PENALTY_NO_OPTS
    
    # Bonus qualità: più alto è lo score, più abbassiamo la fitness (reward)
    # Moltiplichiamo per un fattore piccolo per raffinare
    reward_quality = quality_score_sum * 0.5
    
    total_cost = cost_quantity - reward_quality

    if detailed:
        return {
            "total": total_cost,
            "valid_options": valid_options_count,
            "missing_options": missing_options,
            "blocked_count_debug": pen_block_debug
        }
        
    return total_cost

def cost_offside_avoidance(df_home, obstacles_array, ball_pos, attacking_dir='right', detailed=False):
    """(Offensive) Penalizes players who are offside."""
    home_x = df_home['x'].values
    opp_x = obstacles_array[:, 0]
    offside_dist = 0.0
    
    if len(opp_x) >= 2:
        if attacking_dir == 'right':
            line = np.sort(opp_x)[-2]
            eff_line = max(line, ball_pos[0])
            mask = home_x > eff_line
            if np.any(mask): offside_dist = np.sum(home_x[mask] - eff_line)
        else:
            line = np.sort(opp_x)[1]
            eff_line = min(line, ball_pos[0])
            mask = home_x < eff_line
            if np.any(mask): offside_dist = np.sum(eff_line - home_x[mask])

    if detailed: return {"total": offside_dist, "meters": offside_dist}
    return offside_dist

# --- DEFENSIVE OBJECTIVES ---

def cost_marking(df_home, obstacles_array, detailed=False):
    """
    (Defensive) Ogni avversario deve avere un marcatore vicino.
    Calcola la somma delle distanze minime (Home -> Opponent).
    """
    home = df_home[['x', 'y']].values
    opps = obstacles_array
    
    total_dist = 0.0
    
    # Per ogni avversario, trova il nostro giocatore più vicino
    for opp in opps:
        dists = np.linalg.norm(home - opp, axis=1)
        min_d = np.min(dists)
        total_dist += min_d

    # Normalizziamo un po' (media distanza per avversario)
    avg_marking_dist = total_dist / max(len(opps), 1)
    
    if detailed:
        return {"total": avg_marking_dist, "avg_dist": avg_marking_dist}
    return avg_marking_dist

def cost_defensive_compactness(df_home, detailed=False):
    """
    (Defensive) Minimizza la dispersione dei giocatori dal baricentro.
    """
    coords = df_home[['x', 'y']].values
    centroid = np.mean(coords, axis=0)
    
    # Distanza media dal centroide
    dists = np.linalg.norm(coords - centroid, axis=1)
    dispersion = np.mean(dists)
    
    if detailed:
        return {"total": dispersion, "dispersion": dispersion}
    return dispersion

def cost_defensive_line_height(df_home, ball_pos, detailed=False):
    """
    (Defensive) Premia la linea alta (lontano dalla porta 0), ma non oltre la palla.
    Cost = (1.0 - last_defender_x). Più X è alto, più il costo scende.
    Se X > ball_pos, penalità.
    """
    home_x = df_home['x'].values
    last_defender_x = np.min(home_x)
    
    # Vogliamo massimizzare X (quindi minimizzare 1-X)
    cost = 1.0 - last_defender_x
    
    # Vincolo tattico: non salire ciecamente oltre la palla (se la palla è scoperta)
    # Se last_defender > ball_x, rischio enorme.
    penalty_over_ball = 0.0
    if last_defender_x > ball_pos[0]:
        penalty_over_ball = (last_defender_x - ball_pos[0]) * 5.0
        
    total = cost + penalty_over_ball
    
    if detailed:
        return {"total": total, "line_x": last_defender_x}
    return total

# --- COMMON ---

def cost_ball_pressure(df_home, ball_pos, detailed=False):
    """
    (Defensive/Offensive) Distanza del giocatore più vicino alla palla.
    In difesa è "Pressing", in attacco è "Support".
    """
    dists = np.linalg.norm(df_home[['x', 'y']].values - ball_pos, axis=1)
    min_dist = np.min(dists)
    
    # Fattore di scala per renderlo significativo
    cost = min_dist * 5.0 
    
    if detailed: return {"total": cost, "dist": min_dist}
    return cost