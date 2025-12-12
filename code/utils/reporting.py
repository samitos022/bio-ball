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
    if isinstance(formation_data, pd.DataFrame):
        df = formation_data
    else:
        df = flat_to_formation(formation_data, player_names)

    weights = config.PHASE_WEIGHTS.get(phase_name, config.PHASE_WEIGHTS["Defensive phase"])
    
    print("\n" + "="*100)
    print(f"FITNESS REPORT: {phase_name.upper()}")
    print("="*100)
    print(f"{'OBJECTIVE / DETAIL':<45} | {'RAW VALUE':<12} | {'WEIGHT':<8} | {'FINAL COST':<12}")
    print("-" * 100)

    total_fitness = 0.0

    # --- 1. CONSTRAINTS (Hard) ---
    pos_dict = {"Start": initial_df_ref, "Candidate": df}
    res_constr = penalty_total(pos_dict, detailed=True)
    cost_c = res_constr["total"] * config.OBJ_W_CONSTRAINTS
    total_fitness += cost_c
    
    print(f"\033[1m{'Constraints (Hard)':<45} | {res_constr['total']:<12.4f} | {config.OBJ_W_CONSTRAINTS:<8} | {cost_c:<12.4f}\033[0m")
    if res_constr["total"] > 0.0001:
        if res_constr['boundary'] > 0: print(f"  ├─ Out of Bounds: {res_constr['boundary']:.4f}")
        if res_constr['collision'] > 0: print(f"  ├─ Collisions:    {res_constr['collision']:.4f}")
        if res_constr['transition'] > 0: print(f"  └─ Transition:    {res_constr['transition']:.4f}")

    print("-" * 100)

    # --- 2. OFFENSIVE OBJECTIVES ---
    
    # Coverage
    if weights.get("W_COVERAGE", 0) > 0:
        res = cost_coverage(df, detailed=True)
        cost = res["total"] * weights["W_COVERAGE"]
        total_fitness += cost
        print(f"\033[1m{'Pitch Coverage':<45} | {res['total']:<12.4f} | {weights['W_COVERAGE']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Coverage ratio: {res['coverage_ratio']:.4f}")

    # Passing Lanes
    if weights.get("W_PASSING", 0) > 0:
        res = cost_passing_lanes(df, obstacles, ball_pos, phase_type=phase_name, detailed=True)
        cost = res["total"] * weights["W_PASSING"]
        total_fitness += cost
        
        print(f"\033[1m{'Passing Availability':<45} | {res['total']:<12.4f} | {weights['W_PASSING']:<8} | {cost:<12.4f}\033[0m")
        print(f"  ├─ Quality Score:     {res['quality_score']:.2f} / {res['target_score']} (Target)")
        
        if 'carrier' in res:
            print(f"  ├─ Ball Carrier:      {res['carrier']}")
            
            if res['valid_count'] > 0:
                receivers_str = ", ".join(res['receivers'])
                print(f"  ├─ Valid Receivers:   {receivers_str}")
            else:
                print(f"  ├─ Valid Receivers:   NONE (Isolated!)")
        
        print(f"  └─ Blocked Passes:    {res['blocked_count']}")

    # Offside Avoidance
    if weights.get("W_OFFSIDE", 0) > 0:
        res = cost_offside_avoidance(df, obstacles, ball_pos, detailed=True)
        cost = res["total"] * weights["W_OFFSIDE"]
        total_fitness += cost
        print(f"\033[1m{'Offside Avoidance':<45} | {res['total']:<12.4f} | {weights['W_OFFSIDE']:<8} | {cost:<12.4f}\033[0m")
        if res['total'] > 0:
            print(f"  └─ Meters Offside:    {res['meters']:.4f}")
    
    # Preventive Marking (Counter-attack prevention)
    if weights.get("W_PREV_MARKING", 0) > 0:
        res = cost_preventive_marking(df, obstacles, detailed=True)
        c = res["total"] * weights["W_PREV_MARKING"]
        total_fitness += c
        print(f"\033[1m{'Preventive Marking':<45} | {res['total']:<12.4f} | {weights['W_PREV_MARKING']:<8} | {c:<12.4f}\033[0m")
        print(f"  └─ Threats:           {res['threats']:.4f}")


    # --- 3. DEFENSIVE OBJECTIVES ---

    # Marking
    if weights.get("W_MARKING", 0) > 0:
        res = cost_marking(df, obstacles, detailed=True)
        cost = res["total"] * weights["W_MARKING"]
        total_fitness += cost
        print(f"\033[1m{'Defensive Marking':<45} | {res['total']:<12.4f} | {weights['W_MARKING']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Avg Marker Dist.:  {res['avg_dist']*100:.1f} meters")

    # Compactness
    if weights.get("W_COMPACTNESS", 0) > 0:
        res = cost_defensive_compactness(df, detailed=True)
        cost = res["total"] * weights["W_COMPACTNESS"]
        total_fitness += cost
        print(f"\033[1m{'Defensive Compactness':<45} | {res['total']:<12.4f} | {weights['W_COMPACTNESS']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Dispersion:        {res['dispersion']:.4f}")

    # Line Height
    if weights.get("W_LINE_HEIGHT", 0) > 0:
        res = cost_defensive_line_height(df, ball_pos, detailed=True)
        cost = res["total"] * weights["W_LINE_HEIGHT"]
        total_fitness += cost
        print(f"\033[1m{'Defensive Line Height':<45} | {res['total']:<12.4f} | {weights['W_LINE_HEIGHT']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Last Defender X:   {res['line_x']:.4f} (Target: High)")

    # --- 4. COMMON ---
    
    # Ball Pressure / Support
    if weights.get("W_BALL_PRESS", 0) > 0:
        res = cost_ball_pressure(df, ball_pos, detailed=True)
        cost = res["total"] * weights["W_BALL_PRESS"]
        total_fitness += cost
        label = "Ball Pressing" if "defensive" in phase_name.lower() else "Ball Support"
        print(f"\033[1m{label:<45} | {res['total']:<12.4f} | {weights['W_BALL_PRESS']:<8} | {cost:<12.4f}\033[0m")
        print(f"  └─ Min Dist to Ball:  {res['dist']*100:.1f} meters")

    print("-" * 100)
    print(f"\033[1m{'TOTAL FITNESS':<79} | {total_fitness:<12.4f}\033[0m")
    print("="*100 + "\n")