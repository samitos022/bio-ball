import config
import pandas as pd
from utils.conversion import flat_to_formation
from optimization.constraints import penalty_total
from optimization.cost_functions import (
    cost_coverage, cost_passing_lanes, cost_offside_avoidance,
    cost_marking, cost_defensive_compactness, cost_defensive_line_height,
    cost_ball_pressure, cost_preventive_marking
)

def print_fitness_breakdown(formation_data, player_names, obstacles, ball_pos, initial_df_ref, phase_name):
    """
    Stampa un report dettagliato degli obiettivi attivi per la fase corrente.
    """
    
    if isinstance(formation_data, pd.DataFrame):
        df = formation_data
    else:
        df = flat_to_formation(formation_data, player_names)

    weights = config.PHASE_WEIGHTS.get(phase_name, config.PHASE_WEIGHTS["Fase difensiva"])
    
    print("\n" + "="*100)
    print(f"FITNESS REPORT DETTAGLIATO: {phase_name.upper()}")
    print("="*100)
    print(f"{'OBIETTIVO / DETTAGLIO':<45} | {'VAL GREZZO':<12} | {'PESO':<8} | {'COSTO':<12}")
    print("-" * 100)

    total_fitness = 0.0

    # --- 1. CONSTRAINTS ---
    pos_dict = {"Start": initial_df_ref, "Candidate": df}
    res_constr = penalty_total(pos_dict, detailed=True)
    cost_c = res_constr["total"] * config.OBJ_W_CONSTRAINTS
    total_fitness += cost_c
    
    print(f"\033[1m{'Constraints (Hard)':<45} | {res_constr['total']:<12.4f} | {config.OBJ_W_CONSTRAINTS:<8} | {cost_c:<12.4f}\033[0m")
    if res_constr["total"] > 0.0001:
        if res_constr['boundary'] > 0: print(f"  ├─ Fuori Campo: {res_constr['boundary']:.4f}")
        if res_constr['collision'] > 0: print(f"  ├─ Collisioni:  {res_constr['collision']:.4f}")
        if res_constr['transition'] > 0: print(f"  └─ Transizione: {res_constr['transition']:.4f}")

    print("-" * 100)

    # --- 2. OBIETTIVI OFFENSIVI ---
    
    # Coverage
    if weights.get("W_COVERAGE", 0) > 0:
        res = cost_coverage(df, detailed=True)
        cost = res["total"] * weights["W_COVERAGE"]
        total_fitness += cost
        print(f"\033[1m{'Coverage (Massimizza Area)':<45} | {res['total']:<12.4f} | {weights['W_COVERAGE']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Coverage ratio: {res['coverage_ratio']:.4f}")

    # Passing Lanes (AGGIORNATO PER NUOVE CHIAVI)
    if weights.get("W_PASSING", 0) > 0:
        res = cost_passing_lanes(df, obstacles, ball_pos, detailed=True)
        cost = res["total"] * weights["W_PASSING"]
        total_fitness += cost
        print(f"\033[1m{'Passing Lanes (Bonus Quality)':<45} | {res['total']:<12.4f} | {weights['W_PASSING']:<8} | {cost:<12.4f}\033[0m")
        # Controllo se stiamo usando la nuova versione (dizionario con 'valid_options')
        if 'valid_options' in res:
            print(f"  ├─ Opzioni Valide:    {res['valid_options']} (Target: >=3)")
            print(f"  ├─ Opzioni Mancanti:  {res['missing_options']}")
            print(f"  └─ Blocchi ignorati:  {res['blocked_count_debug']}")
        else:
            # Fallback vecchia versione
            print(f"  └─ Valore: {res['total']}")

    # Offside Avoidance
    if weights.get("W_OFFSIDE", 0) > 0:
        res = cost_offside_avoidance(df, obstacles, ball_pos, detailed=True)
        cost = res["total"] * weights["W_OFFSIDE"]
        total_fitness += cost
        print(f"\033[1m{'Offside Avoidance':<45} | {res['total']:<12.4f} | {weights['W_OFFSIDE']:<8} | {cost:<12.4f}\033[0m")
        if res['total'] > 0:
            print(f"  └─ Metri oltre la linea: {res['meters']:.4f}")
    
    if weights.get("W_PREV_MARKING", 0) > 0:
        res = cost_preventive_marking(df, obstacles, detailed=True)
        c = res["total"] * weights["W_PREV_MARKING"]
        total_fitness += c
        print(f"\033[1m{'Preventive Marking':<45} | {res['total']:<12.4f} | {weights['W_PREV_MARKING']:<8} | {c:<12.4f}\033[0m")
        print(f"  └─ Threats: {res['threats']:.4f}")


    # --- 3. OBIETTIVI DIFENSIVI ---

    # Marking
    if weights.get("W_MARKING", 0) > 0:
        res = cost_marking(df, obstacles, detailed=True)
        cost = res["total"] * weights["W_MARKING"]
        total_fitness += cost
        print(f"\033[1m{'Marking (Distanza Avversari)':<45} | {res['total']:<12.4f} | {weights['W_MARKING']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Distanza media marcatore: {res['avg_dist']*100:.1f} metri")

    # Compactness
    if weights.get("W_COMPACTNESS", 0) > 0:
        res = cost_defensive_compactness(df, detailed=True)
        cost = res["total"] * weights["W_COMPACTNESS"]
        total_fitness += cost
        print(f"\033[1m{'Defensive Compactness':<45} | {res['total']:<12.4f} | {weights['W_COMPACTNESS']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Dispersione dal centro: {res['dispersion']:.4f}")

    # Line Height
    if weights.get("W_LINE_HEIGHT", 0) > 0:
        res = cost_defensive_line_height(df, ball_pos, detailed=True)
        cost = res["total"] * weights["W_LINE_HEIGHT"]
        total_fitness += cost
        print(f"\033[1m{'Defensive Line Height':<45} | {res['total']:<12.4f} | {weights['W_LINE_HEIGHT']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ X Ultimo Difensore: {res['line_x']:.4f} (Target: Alto)")

    # --- 4. COMMON ---
    
    if weights.get("W_BALL_PRESS", 0) > 0:
        res = cost_ball_pressure(df, ball_pos, detailed=True)
        cost = res["total"] * weights["W_BALL_PRESS"]
        total_fitness += cost
        label = "Ball Pressing" if "difensiva" in phase_name.lower() else "Ball Support"
        print(f"\033[1m{label:<45} | {res['total']:<12.4f} | {weights['W_BALL_PRESS']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Distanza min dalla palla: {res['dist']*100:.1f} metri")

    print("-" * 100)
    print(f"\033[1m{'TOTALE FITNESS':<79} | {total_fitness:<12.4f}\033[0m")
    print("="*100 + "\n")