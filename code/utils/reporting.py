import config
import pandas as pd
import numpy as np
from utils.conversion import flat_to_formation
from optimization.constraints import penalty_total
from optimization.cost_functions import (
    cost_coverage, 
    cost_passing_lanes, 
    cost_ball_support, 
    cost_offside
)

def print_fitness_breakdown(formation_data, player_names, obstacles, ball_pos, initial_df_ref):
    """
    Stampa una tabella dettagliata con i valori grezzi, pesati e i sotto-dettagli.
    """
    
    # 1. Gestione Input
    if isinstance(formation_data, pd.DataFrame):
        df_candidate = formation_data
    else:
        df_candidate = flat_to_formation(formation_data, player_names)

    # 2. Calcolo Obiettivi (DETAILED = True)
    res_cover = cost_coverage(df_candidate, detailed=True)
    res_pass  = cost_passing_lanes(df_candidate, obstacles, ball_pos, detailed=True)
    res_ball  = cost_ball_support(df_candidate, ball_pos, detailed=True)
    res_off   = cost_offside(df_candidate, obstacles, ball_pos, detailed=True)

    # 3. Calcolo Constraints (DETAILED = True)
    pos_dict = {"Start": initial_df_ref, "Candidate": df_candidate}
    res_constr = penalty_total(pos_dict, detailed=True)

    # 4. Calcolo Costi Pesati
    w_const = config.OBJ_W_CONSTRAINTS
    w_cover = config.OBJ_W_COVER
    w_pass  = config.OBJ_W_PASS
    w_ball  = config.OBJ_W_BALL
    w_off   = config.OBJ_W_OFFSIDE

    total_fitness = (res_constr["total"] * w_const) + \
                    (res_cover["total"] * w_cover) + \
                    (res_pass["total"] * w_pass) + \
                    (res_ball["total"] * w_ball) + \
                    (res_off["total"] * w_off)

    # 5. Stampa Tabella Formattata
    print("\n" + "="*95)
    print(f"{'FITNESS BREAKDOWN DETTAGLIATO':^95}")
    print("="*95)
    print(f"{'OBIETTIVO / DETTAGLIO':<40} | {'VAL GREZZO':<12} | {'PESO':<10} | {'COSTO FINALE':<12}")
    print("-" * 95)
    
    # --- CONSTRAINTS ---
    c_tot_constr = res_constr["total"] * w_const
    print(f"\033[1m{'Constraints (Hard)':<40} | {res_constr['total']:<12.4f} | {w_const:<10.1f} | {c_tot_constr:<12.4f}\033[0m")
    if res_constr["total"] > 0:
        print(f"  ├─ Fuori Campo: {res_constr['boundary']:.2f}")
        print(f"  ├─ Collisioni:  {res_constr['collision']:.2f}")
        print(f"  ├─ Ordine:      {res_constr['order']:.2f}")
        print(f"  └─ Transizione: {res_constr['transition']:.2f}")

    # --- COVERAGE ---
    c_tot_cover = res_cover["total"] * w_cover
    print(f"-" * 95)
    print(f"\033[1m{'Coverage':<40} | {res_cover['total']:<12.4f} | {w_cover:<10.1f} | {c_tot_cover:<12.4f}\033[0m")
    print(f"  ├─ Area Coperta: {res_cover['raw_area']:.4f} (su 1.0)")
    print(f"  └─ Spazio Vuoto: {res_cover['empty_space']:.4f}")

    # --- PASSING LANES ---
    c_tot_pass = res_pass["total"] * w_pass
    print(f"-" * 95)
    print(f"\033[1m{'Passing Lanes':<40} | {res_pass['total']:<12.4f} | {w_pass:<10.1f} | {c_tot_pass:<12.4f}\033[0m")
    print(f"  ├─ Passaggi Validi: {res_pass['num_valid']}")
    print(f"  ├─ Passaggi Bloccati: {res_pass['num_blocked']} (Pen: {res_pass['p_block']:.2f})")
    print(f"  ├─ Penalità Angolo: {res_pass['p_angle']:.2f}")
    print(f"  └─ Penalità Lunghezza: {res_pass['p_long']:.2f}")

    # --- BALL SUPPORT ---
    c_tot_ball = res_ball["total"] * w_ball
    print(f"-" * 95)
    print(f"\033[1m{'Ball Support':<40} | {res_ball['total']:<12.4f} | {w_ball:<10.1f} | {c_tot_ball:<12.4f}\033[0m")
    print(f"  └─ Distanza min (m): {res_ball['min_distance_meters']*100:.2f} m") # Assumendo campo unitario 100m

    # --- OFFSIDE ---
    c_tot_off = res_off["total"] * w_off
    print(f"-" * 95)
    print(f"\033[1m{'Offside':<40} | {res_off['total']:<12.4f} | {w_off:<10.1f} | {c_tot_off:<12.4f}\033[0m")
    if res_off["players_offside"] > 0:
        print(f"  ├─ Giocatori in fuorigioco: {res_off['players_offside']}")
        print(f"  └─ Metri totali oltre: {res_off['total_meters']:.4f}")
    else:
        print(f"  └─ Nessun fuorigioco")

    print("=" * 95)
    print(f"\033[1m{'TOTALE FITNESS':<77} | {total_fitness:<12.4f}\033[0m")
    print("=" * 95 + "\n")