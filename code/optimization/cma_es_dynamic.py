import cma
import numpy as np
from utils.animation_dynamic import interpolate_vectors, save_generation_plot
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function
from utils.away_reaction import react_away_to_home
import config

def run_optimization(initial_guess, initial_away_df, ball_position, player_names, phase_name="Defensive phase"):

    initial_df_ref = flat_to_formation(initial_guess, player_names)
   
    fitness_args = (player_names, initial_away_df, ball_position, initial_df_ref, phase_name, 'dynamic')
    
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

    print(f"Starting DYNAMIC optimization ({phase_name})...")
    
    prev_best_vector = initial_guess.copy()
    SAVE_EVERY = 5
    INTERPOLATION_STEPS = 4

    while not es.stop():
        solutions = es.ask()
        fitness_values = []
        
        best_vec_gen = None
        best_cost_gen = float('inf')
        best_df_away_gen = None

        for x in solutions:
            f = objective_function(x, fitness_args)
            fitness_values.append(f)
            
            if f < best_cost_gen:
                best_cost_gen = f
                best_vec_gen = x
                df_home = flat_to_formation(x, player_names)
                best_df_away_gen = react_away_to_home(df_home, initial_away_df, ball_position)

        es.tell(solutions, fitness_values)
        cost_history.append(best_cost_gen)

        if es.countiter % SAVE_EVERY == 0 and best_vec_gen is not None:
            inter_frames = interpolate_vectors(prev_best_vector, best_vec_gen, steps=INTERPOLATION_STEPS)
            for f in inter_frames:
                save_generation_plot(f, player_names, best_df_away_gen, ball_position, es.countiter)
            prev_best_vector = best_vec_gen.copy()

        if es.countiter % 10 == 0:
            print(f"Gen {es.countiter}: best cost = {best_cost_gen:.4f}")

    best_vector = es.result.xbest
    return best_vector, cost_history