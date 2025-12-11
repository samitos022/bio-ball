import numpy as np
import pandas as pd
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

def exclude_goalkeeper(df_home):
    """Rimuove il portiere (giocatore con X minima)."""
    if isinstance(df_home, pd.DataFrame):
        gk_idx = df_home['x'].idxmin()
        return df_home.drop(index=gk_idx)
    min_x_idx = np.argmin(df_home[:, 0])
    return np.delete(df_home, min_x_idx, axis=0)

def cost_coverage(df, detailed=False):
    """
    Copertura campo con pesi tattici graduali.
    Zone centrali-avanzate = massimo valore
    Fasce laterali = valore medio
    Zone difensive centrali/angoli = minimo valore
    """
    df_outfield = exclude_goalkeeper(df)
    points = df_outfield[['x', 'y']].values
    
    if len(points) < 3:
        return {"total": 100.0, "coverage": 0.0} if detailed else 100.0
    
    # Parametri
    grid_size = 20
    player_radius = 0.12
    
    # Crea griglia
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size)
    )
    cells = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    
    # CALCOLO PESI TATTICI (vettorizzato)
    x, y = cells[:, 0], cells[:, 1]
    
    # 1. Avanzamento: lineare crescente (0.5 → 1.5)
    advancement = 0.5 + x
    
    # 2. Centralità: gaussiana (max al centro y=0.5, min ai lati)
    centrality = np.exp(-5 * (y - 0.5)**2)  # Picco 1.0 al centro, ~0.3 ai bordi
    
    # 3. Penalità angoli difensivi (x<0.3 e y estreme)
    corner_penalty = np.where((x < 0.3) & ((y < 0.2) | (y > 0.8)), 0.3, 1.0)
    
    # Peso finale combinato
    weights = advancement * centrality * corner_penalty
    
    # Check copertura
    covered = np.zeros(len(cells), dtype=bool)
    for point in points:
        distances = np.linalg.norm(cells - point, axis=1)
        covered |= (distances <= player_radius)
    
    # Metriche
    weighted_coverage = np.sum(weights[covered])
    max_coverage = np.sum(weights)
    coverage_ratio = weighted_coverage / max_coverage
    
    # Costo: penalizza bassa copertura
    cost = (1.0 - coverage_ratio) * 3
    
    if detailed:
        return {
            "total": cost,
            "coverage_ratio": np.sum(covered) / len(cells),
            "weighted_ratio": coverage_ratio,
            "avg_x": np.mean(points[:, 0])
        }
    
    return cost

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
    (Defensive) Ogni avversario DI MOVIMENTO deve avere un marcatore vicino.
    Escludiamo il portiere avversario dal calcolo per evitare pressing inutile.
    """
    home = df_home[['x', 'y']].values
    opps = obstacles_array
    
    # --- MODIFICA: ESCLUSIONE PORTIERE ---
    # Assumiamo che il portiere avversario sia quello con la X più alta
    # (più lontano dalla nostra porta che è a X=0)
    if len(opps) > 0:
        gk_idx = np.argmax(opps[:, 0]) # Indice del giocatore con X max
        # Rimuoviamo il portiere dalla lista degli avversari da marcare
        targets = np.delete(opps, gk_idx, axis=0)
    else:
        targets = opps
        
    sum_sq_dist = 0.0
    
    # Se non ci sono avversari (caso limite), costo 0
    if len(targets) == 0:
        if detailed: return {"total": 0.0, "avg_dist": 0.0}
        return 0.0
    
    # Per ogni avversario di movimento, trova il nostro giocatore più vicino
    for opp in targets:
        dists = np.linalg.norm(home - opp, axis=1)
        min_d = np.min(dists)
        
        # Penalità Quadratica (per punire severamente l'uomo lasciato solo)
        sum_sq_dist += min_d ** 2

    # Normalizziamo sul numero di avversari effettivi (es. 10)
    cost = sum_sq_dist / len(targets)
    
    # Scaling per rendere il numero rilevante per l'ottimizzatore
    cost_scaled = cost * 10.0
    
    if detailed:
        # RMSE in metri reali
        rmse = np.sqrt(sum_sq_dist / len(targets))
        return {"total": cost_scaled, "avg_dist": rmse}
    
    return cost_scaled

def cost_defensive_compactness(df_home, detailed=False):
    """(Defensive) Compattezza (Escluso Portiere)."""
    df_outfield = exclude_goalkeeper(df_home)
    coords = df_outfield[['x', 'y']].values
    
    centroid = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    dispersion = np.mean(dists)
    
    final_cost = dispersion * 5.0
    
    if detailed:
        return {"total": final_cost, "dispersion": dispersion}
    return final_cost

def cost_defensive_line_height(df_home, ball_pos, detailed=False):
    """
    (Defensive) Altezza Linea Dinamica.
    La difesa deve salire accompagnando la palla, ma senza farsi scavalcare.
    """
    # 1. Trova l'ultimo uomo
    df_outfield = exclude_goalkeeper(df_home)
    outfield_x = df_outfield['x'].values
    if len(outfield_x) == 0: return 0.0
    
    last_man_x = np.min(outfield_x)
    ball_x = ball_pos[0]

    # 2. Definisci il Target Dinamico
    # Vogliamo stare dietro la palla di un cuscinetto di sicurezza (es. 5-10 metri)
    # Ma non possiamo salire oltre il centrocampo (0.5) per non rischiare troppo i lanci lunghi
    safety_buffer = 0.20  # 20 metri dietro la palla
    target_x = min(ball_x - safety_buffer, 0.55) 
    
    # Non possiamo nemmeno difendere dietro la linea di porta (0.0)
    target_x = max(target_x, 0.05) 

    # 3. Calcolo Costo
    cost = 0.0
    
    # A. Siamo troppo bassi rispetto al target? (Squadra lunga)
    # Esempio: Palla a centrocampo, noi siamo in area.
    if last_man_x < target_x:
        cost += (target_x - last_man_x) * 5.0  # Spingiamo su la linea
        
    # B. Siamo troppo alti? (Rischio imbucata / Siamo stati superati)
    # Se siamo oltre la palla, disastro.
    if last_man_x > ball_x:
        cost += (last_man_x - ball_x) * 50.0  # Penalità enorme (scappare indietro!)
    
    # C. Siamo tra la palla e il target? (Zona grigia accettabile)
    # Se last_man_x è tra (ball_x - buffer) e ball_x, va bene, stiamo accorciando.

    if detailed:
        return {
            "total": cost,
            "line_x": last_man_x,
            "target_dynamic": target_x,
            "ball_x": ball_x
        }
        
    return cost

def cost_preventive_marking(df_home, obstacles_array, detailed=False):
    """(Possesso Offensivo) Marcatura Preventiva."""
    home = df_home[['x', 'y']].values
    threats = [opp for opp in obstacles_array if opp[0] < 0.5]
    
    if not threats: 
        if detailed: return {"total": 0.0, "threats": 0}
        return 0.0
    
    cost = 0.0
    for threat in threats:
        dists = np.linalg.norm(home - threat, axis=1)
        min_d = np.min(dists)
        cost += min_d
        
    final_cost = cost * 5.0
    
    if detailed:
        return {"total": final_cost, "threats": len(threats), "raw_dist": cost}
    return final_cost

# --- COMMON ---
def cost_ball_pressure(df_home, ball_pos, detailed=False):
    """
    (Defensive/Offensive) Distanza del giocatore più vicino alla palla.
    In difesa è "Pressing", in attacco è "Support".
    """
    dists = np.linalg.norm(df_home[['x', 'y']].values - ball_pos, axis=1)
    min_dist = np.min(dists)
    
    # Fattore di scala per renderlo significativo
    cost = min_dist * 20.0 
    
    if detailed: return {"total": cost, "dist": min_dist}
    return cost

def cost_passing_lanes(df_home, obstacles_array, ball_pos, phase_type="Possesso offensivo", detailed=False):
    # ... (parte iniziale invariata) ...
    home = df_home[['x', 'y']].values
    player_names = df_home.index.tolist() # <--- RECUPERIAMO I NOMI
    n = len(home)
    
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier_idx = np.argmin(dists) 
    p1 = home[ball_carrier_idx]
    ball_carrier_name = player_names[ball_carrier_idx] # <--- NOME PORTATORE

    total_quality_score = 0.0
    valid_options_count = 0
    blocked_count = 0
    
    valid_receivers = [] # <--- LISTA PER I NOMI

    # Parametri specifici per fase
    is_offensive = "offensivo" in phase_type.lower()
    target_score = config.PASS_TARGET_SCORE_OFF if is_offensive else config.PASS_TARGET_SCORE_DEF

    for j in range(n):
        if j == ball_carrier_idx: continue
        p2 = home[j]
        receiver_name = player_names[j] # <--- NOME RICEVITORE
        
        # 1. Check Blocco
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            blocked_count += 1
            continue 
            
        # 2. Check Lunghezza/Vicinanza
        dist = np.linalg.norm(p1 - p2)
        if dist > config.PASS_MAX_LEN: continue
        if dist < 0.05: continue

        # Calcolo Score
        pass_val = 1.0
        if is_offensive:
            ang = angle_score(p1, p2)
            angle_multiplier = 1.0 + (ang * 0.5) 
            pass_val *= angle_multiplier
        else:
            if dist > 0.20: pass_val *= 0.8

        total_quality_score += pass_val
        valid_options_count += 1
        
        # AGGIUNGIAMO ALLA LISTA (Nome + Score formattato)
        valid_receivers.append(f"{receiver_name} ({pass_val:.1f})")

    # ... (Calcolo costo finale invariato) ...
    cost = config.PASS_PENALTY_NO_OPTS * np.exp(-total_quality_score / target_score)

    if detailed:
        return {
            "total": cost,
            "quality_score": total_quality_score,
            "target_score": target_score,
            "valid_count": valid_options_count,
            "blocked_count": blocked_count,
            "carrier": ball_carrier_name,      # <--- RESTITUIAMO CHI HA LA PALLA
            "receivers": valid_receivers       # <--- RESTITUIAMO LA LISTA
        }
        
    return cost