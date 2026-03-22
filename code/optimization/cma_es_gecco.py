import cma
import numpy as np
from utils.conversion import flat_to_formation
from optimization.objectives_gecco import objective_function_gecco
import config

def run_optimization_gecco(initial_guess, initial_away_df, ball_position, player_names, phase_name="Defensive phase"):
    
    initial_df_ref = flat_to_formation(initial_guess, player_names)
    
    fitness_args = (player_names, initial_away_df[["x", "y"]].values, ball_position, initial_df_ref, phase_name, 'static')
    
    sigma_init = config.CMA_SIGMA_INIT
    options = {
        'maxiter': config.CMA_MAXITER,
        'popsize': config.CMA_POPSIZE,
        'bounds': [0, 1], 
        'verbose': -1,
        'tolfun': config.CMA_TOLFUN
    }

    es = cma.CMAEvolutionStrategy(initial_guess, sigma_init, options)
    cost_history =[]

    print(f"Starting GECCO DYNAMIC optimization ({phase_name})...")

    while not es.stop():
        solutions = es.ask()
        fitness_values =[objective_function_gecco(x, fitness_args) for x in solutions]
        es.tell(solutions, fitness_values)
        
        current_best = min(fitness_values)
        cost_history.append(current_best)

        if es.countiter % 10 == 0:
            best_idx = np.argmin(fitness_values)
            best_candidate = solutions[best_idx]
            
            detailed_args = fitness_args + (True,)
            stats = objective_function_gecco(best_candidate, detailed_args)
            
            print(f"Gen {es.countiter} | FIT: {stats['FITNESS']:.1f} | "
                  f"Att: {stats['Attack_Reward']:.1f} | "
                  f"Risk: +{stats['Counter_Risk']:.1f} | "
                  f"Def: {stats['Def_Reward']:.1f} | "
                  f"Ghost: +{stats['Ghost_Penalty']:.1f} | "
                  f"Pen: +{stats['Hard_Penalties']['Total']:.1f}")
                  
            if stats['Hard_Penalties']['Total'] > 1.0:
                p = stats['Hard_Penalties']
                print(f"    └-> [Violations] Out: {p['Boundaries']:.1f}, GK: {p['Goalkeeper']:.1f}, "
                      f"Collisions: {p['Collisions']:.1f}, Offside: {p['Offside']:.1f}, Ball: {p['Ball_Action']:.1f}")

    best_vector = es.result.xbest
    return best_vector, cost_history