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
    (Defensive) Altezza Linea Dinamica ed Elastica.
    
    Novità:
    1. Calcola la linea basandosi sulla MEDIA degli ultimi 3 difensori (non solo 1).
    2. La distanza ideale dalla palla varia: 
       - Palla vicina alla porta = Difensori vicini alla palla (densità).
       - Palla lontana = Difensori più staccati (copertura profondità).
    """
    # 1. Seleziona i difensori (escludendo il portiere)
    # Assumiamo che df_outfield contenga già solo i giocatori di movimento
    # Se non hai la funzione esterna, filtro qui veloce per sicurezza (escludo chi è troppo vicino a 0.0 se c'è un GK)
    outfield_x = exclude_goalkeeper(df_home)['x'].values
    # (Se hai una funzione exclude_goalkeeper usala qui: outfield_x = exclude_goalkeeper(df_home)['x'].values)
    
    if len(outfield_x) < 3: 
        # Fallback se ci sono meno di 3 giocatori (es. espulsioni o test)
        line_x = np.min(outfield_x) if len(outfield_x) > 0 else 0.0
    else:
        # Ordina le posizioni X (dal più vicino alla porta 0.0 al più lontano)
        sorted_x = np.sort(outfield_x)
        # Prende i primi 3 (i 3 più arretrati) e fa la media
        line_x = np.mean(sorted_x[:1])

    ball_x = ball_pos[0]

    # 2. Calcolo Distanza Ideale Dinamica (Elasticità)
    # Base fissa (minimo sindacale): 0.05 (5 metri)
    # Scaling factor: più la palla è alta, più spazio lasciamo dietro la palla.
    # Formula: 5m + (25% della posizione palla)
    dynamic_buffer = 0.05 + (ball_x * 0.25)
    
    # Calcolo dove dovrebbe essere la linea (Palla - Buffer)
    target_x = ball_x - dynamic_buffer
    
    # Vincoli di campo (non possiamo difendere dentro la porta o oltre il centrocampo difensivo puro)
    target_x = max(target_x, 0.05) # Mai schiacciarsi sulla linea di porta
    target_x = min(target_x, 0.60) # Mai salire scriteriatamente oltre centrocampo

    # 3. Calcolo Costo
    cost = 0.0
    
    # A. Penalità "Too Deep" (Siamo troppo bassi rispetto al target?)
    # Se dobbiamo stare a 30m e siamo a 15m, spingiamo su.
    if line_x < target_x:
        # Moltiplicatore progressivo: piccolo errore pesa poco, grande errore pesa tanto
        cost += (target_x - line_x) * 8.0 
        
    # B. Penalità "Over Ball" (La linea media ha superato la palla?)
    # Attenzione: se la media dei 3 difensori è OLTRE la palla, è gravissimo (buco centrale).
    if line_x > ball_x:
        cost += (line_x - ball_x) * 50.0  # Panic penalty: SCAPPARE INDIETRO!

    if detailed:
        return {
            "total": cost,
            "line_x": line_x,          # Dove siamo (media ultimi 3)
            "target_dynamic": target_x,# Dove dovremmo essere
            "ideal_dist": dynamic_buffer, # Cuscinetto calcolato
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
    """
    Calcola il costo dei passaggi usando una LOGICA IBRIDA (Saturazione + Bonus).
    
    1. Saturazione (Esponenziale): Punisce severamente se non si raggiunge il 'Target Score'.
       Serve a garantire il minimo sindacale di opzioni.
    2. Bonus (Lineare): Premia ulteriormente per ogni punto di qualità aggiunto.
       Serve a spingere l'ottimizzatore a cercare l'eccellenza e non accontentarsi.
    """
    home = df_home[['x', 'y']].values
    player_names = df_home.index.tolist() # Recuperiamo i nomi per il report
    n = len(home)
    
    # Identifica portatore
    dists = np.linalg.norm(home - ball_pos, axis=1)
    ball_carrier_idx = np.argmin(dists) 
    p1 = home[ball_carrier_idx]
    ball_carrier_name = player_names[ball_carrier_idx]

    total_quality_score = 0.0
    valid_options_count = 0
    blocked_count = 0
    
    valid_receivers = [] # Lista per il report dettagliato
    
    # Parametri specifici per fase
    is_offensive = "offensivo" in phase_type.lower()
    target_score = config.PASS_TARGET_SCORE_OFF if is_offensive else config.PASS_TARGET_SCORE_DEF

    for j in range(n):
        if j == ball_carrier_idx: continue
        p2 = home[j]
        receiver_name = player_names[j]
        
        # 1. Check Blocco (Ostacoli)
        block = False
        for opp in obstacles_array:
            if point_line_distance(opp, p1, p2) < config.PASS_BLOCK_THRESHOLD:
                block = True
                break
        
        if block:
            blocked_count += 1
            continue # Passaggio bloccato -> Score 0
            
        # 2. Valutazione Qualità Passaggio (Se libero)
        dist = np.linalg.norm(p1 - p2)
        
        # Filtro lunghezza eccessiva
        if dist > config.PASS_MAX_LEN:
            continue
            
        # Filtro "Ammucchiata" (Importante per l'attacco!)
        # Se sei troppo vicino (< 5m), non sei un'opzione utile, dai solo fastidio.
        if dist < 0.05: 
            continue

        # Calcolo Score Base
        pass_val = 1.0
        
        if is_offensive:
            # FASE OFFENSIVA: Premia la verticalità
            # angle_score va da -1 (dietro) a 1 (avanti)
            # Trasformiamo in moltiplicatore: 0.5 (dietro) a 1.5 (avanti)
            ang = angle_score(p1, p2)
            angle_multiplier = 1.0 + (ang * 0.5) 
            pass_val *= angle_multiplier
            
        else:
            # FASE DIFENSIVA: Premia la sicurezza
            # Se il passaggio è medio-lungo (es. > 20m), vale un po' meno per il rischio
            if dist > 0.20: 
                pass_val *= 0.8

        total_quality_score += pass_val
        valid_options_count += 1
        
        # Salviamo nome e score per il report
        valid_receivers.append(f"{receiver_name} ({pass_val:.1f})")

    # --- 3. Calcolo Costo Finale Ibrido ---
    
    # A. Componente di Sicurezza (Esponenziale)
    # Punisce molto se siamo sotto il target, diventa quasi 0 se siamo sopra.
    saturation_cost = config.PASS_PENALTY_NO_OPTS * np.exp(-total_quality_score / target_score)
    
    # B. Componente di Eccellenza (Bonus Lineare)
    # Sottrae costo (dà bonus negativo) per ogni punto di qualità accumulato.
    # Spinge a migliorare anche dopo aver raggiunto il target.
    # Fattore 0.5: bilanciato per non ignorare totalmente la Coverage.
    reward_factor = 0.5 
    linear_bonus = total_quality_score * reward_factor
    
    # Costo Totale = Paura di sbagliare - Voglia di fare bene
    total_cost = saturation_cost - linear_bonus

    if detailed:
        return {
            "total": total_cost,
            "quality_score": total_quality_score,
            "target_score": target_score,
            "valid_count": valid_options_count,
            "blocked_count": blocked_count,
            "carrier": ball_carrier_name,
            "receivers": valid_receivers,
            # Info di debug utili
            "cost_sat": saturation_cost,
            "cost_bonus": -linear_bonus
        }
        
    return total_cost