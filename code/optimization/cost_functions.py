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
    """
    Rimuove il portiere dalla formazione (assumendo che sia il giocatore con X minima).
    Restituisce un DataFrame o array con 10 giocatori.
    """
    # Se il DataFrame ha le colonne x, y, troviamo l'indice con x minimo
    if isinstance(df_home, pd.DataFrame):
        # Assumiamo che il portiere sia quello più vicino alla linea di fondo (x=0)
        # Nota: questo funziona se attacchiamo verso destra (x=1) e difendiamo x=0.
        gk_idx = df_home['x'].idxmin()
        return df_home.drop(index=gk_idx)
    
    # Se è un array numpy
    min_x_idx = np.argmin(df_home[:, 0])
    return np.delete(df_home, min_x_idx, axis=0)

# --- OFFENSIVE OBJECTIVES ---

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

# =============================================================================
# 1. FUNZIONI OFFENSIVE
# =============================================================================

import numpy as np

def cost_coverage(df, detailed=False):
    """
    Copertura campo basata su griglia semplice.
    Divide il campo in celle e conta quante sono coperte dai giocatori.
    
    MOLTO PIÙ EFFICACE del ConvexHull perché:
    - Valuta distribuzione interna
    - Premia copertura zone strategiche
    - Penalizza buchi nella formazione
    """
    df_outfield = exclude_goalkeeper(df)
    points = df_outfield[['x', 'y']].values
    
    if len(points) < 3:
        if detailed: return {"total": 100.0, "coverage": 0.0}
        return 100.0
    
    # --- PARAMETRI ---
    grid_size = 20                    # Griglia 20x20 (400 celle)
    player_radius = 0.12              # Raggio influenza giocatore (12m su campo 100m)
    
    # Crea griglia
    x_cells = np.linspace(0, 1, grid_size)
    y_cells = np.linspace(0, 1, grid_size)
    
    covered_count = 0
    weighted_coverage = 0.0
    max_weight = 0.0
    
    # Per ogni cella, verifica se è coperta
    for x in x_cells:
        for y in y_cells:
            cell_pos = np.array([x, y])
            
            # Peso cella (zone importanti valgono di più)
            weight = 1.0
            
            # Terzo offensivo (x > 0.6): +50%
            if x > 0.6:
                weight = 1.5
            
            # Corridoio centrale (y tra 0.3-0.7): +50%
            if 0.3 < y < 0.7:
                weight *= 1.5
            
            max_weight += weight
            
            # Check copertura: almeno 1 giocatore vicino?
            distances = np.linalg.norm(points - cell_pos, axis=1)
            min_dist = np.min(distances)
            
            if min_dist <= player_radius:
                covered_count += 1
                weighted_coverage += weight
    
    # Metriche
    total_cells = grid_size * grid_size
    coverage_ratio = covered_count / total_cells
    weighted_ratio = weighted_coverage / max_weight
    
    # Calcolo avanzamento medio (bonus per giocare alto)
    avg_x = np.mean(points[:, 0])
    advancement_bonus = avg_x * 0.3
    
    # COSTO FINALE (minimizza)
    # Vogliamo: alta copertura + zone strategiche + avanzamento
    cost = (
        (1.0 - weighted_ratio) * 5.0 -    # Penalità bassa copertura
        advancement_bonus                  # Bonus giocare alto
    )
    
    if detailed:
        return {
            "total": cost,
            "coverage_ratio": coverage_ratio,
            "weighted_ratio": weighted_ratio,
            "covered_cells": covered_count,
            "total_cells": total_cells,
            "avg_x": avg_x
        }
    
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

def cost_defensive_line_height(df_home, ball_pos, optimal_height=0.3, detailed=False):
    """(Defensive) Altezza Linea (Escluso Portiere)."""
    df_outfield = exclude_goalkeeper(df_home)
    outfield_x = df_outfield['x'].values
    
    if len(outfield_x) == 0: 
        if detailed: return {"total": 0.0, "line_x": 0.0}
        return 0.0
    
    last_man_x = np.min(outfield_x)
    
    too_deep_penalty = 0.0
    if last_man_x < optimal_height: 
        too_deep_penalty = (optimal_height - last_man_x) * 2.0
        
    over_ball_penalty = 0.0
    if last_man_x > ball_pos[0]:
        over_ball_penalty = (last_man_x - ball_pos[0]) * 10.0
        
    total = too_deep_penalty + over_ball_penalty
    
    if detailed:
        return {"total": total, "line_x": last_man_x, "opt": optimal_height}
    return total

def cost_preventive_marking(df_home, obstacles_array, detailed=False):
    """(Possesso Offensivo) Marcatura Preventiva."""
    home = df_home[['x', 'y']].values
    threats = [opp for opp in obstacles_array if opp[0] < 0.4]
    
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
    cost = min_dist * 5.0 
    
    if detailed: return {"total": cost, "dist": min_dist}
    return cost