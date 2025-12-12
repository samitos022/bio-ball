import cma
import numpy as np
from utils.animation import save_generation_plot
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function 
import config

def run_optimization(initial_guess, obstacles, ball_position, player_names, phase_name="Defensive phase"):
    """
    CMA-ES Static Opponent.
    """
    
    initial_df_ref = flat_to_formation(initial_guess, player_names)
    fitness_args = (player_names, obstacles, ball_position, initial_df_ref, phase_name, 'static')
    
    # 2. Configurazione CMA
    sigma_init = config.CMA_SIGMA_INIT
    options = {
        'maxiter': config.CMA_MAXITER,
        'popsize': config.CMA_POPSIZE,
        'bounds': [0, 1], 
        'verbose': -1,
        'tolfun': config.CMA_TOLFUN
    }

    es = cma.CMAEvolutionStrategy(initial_guess, sigma_init, options)
    cost_history = []

    print(f"Starting STATIC optimization ({phase_name})...")

    while not es.stop():
        solutions = es.ask()
        fitness_values = [objective_function(x, fitness_args) for x in solutions]
        es.tell(solutions, fitness_values)
        
        current_best = min(fitness_values)
        cost_history.append(current_best)
        
        idx_best = np.argmin(fitness_values)
        best_vector_gen = solutions[idx_best]

        if es.countiter % 5 == 0:
            save_generation_plot(
                best_vector_gen, player_names, obstacles, ball_position, es.countiter
            )

        if es.countiter % 10 == 0:
            print(f"Gen {es.countiter}: Cost {current_best:.4f}")

    result = es.result
    best_vector = result.xbest
    
    return best_vector, cost_history