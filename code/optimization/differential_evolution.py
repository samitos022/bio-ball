import numpy as np
from scipy.optimize import differential_evolution
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function
from utils.away_reaction import react_away_to_home
import config


def run_de_optimization(initial_guess, initial_away_df, ball_position, player_names):
    dim = len(initial_guess)

    bounds = [(0, 1)] * dim

    history = []
    step_counter = 0

    def fitness(x, player_names, initial_away_df, ball_pos, df_ref):
        df_candidate = flat_to_formation(x, player_names)

        away_df = react_away_to_home(
            home_df=df_candidate,
            base_away_df=initial_away_df,
            ball_pos=ball_pos
        )

        args = (player_names, initial_away_df, ball_pos, df_ref, 'dynamic')
        
        return objective_function(x, args)

    df_ref = flat_to_formation(initial_guess, player_names)

    def callbackF(xk, convergence):
        nonlocal step_counter
        step_counter += 1

        cost = fitness(xk, player_names, initial_away_df, ball_position, df_ref)
        history.append(cost)

        if step_counter % 10 == 0:
            print(f"DE fitness {step_counter}: f(x)={cost}")

        return False

    result = differential_evolution(
        func=lambda x: fitness(x, player_names, initial_away_df, ball_position, df_ref),
        bounds=bounds,
        maxiter=config.DE_MAXITER,
        popsize=config.DE_POPSIZE,
        mutation=(config.DE_MUTATION[0], config.DE_MUTATION[1]),
        recombination=config.DE_RECOMBINATION,
        callback=callbackF,
        polish=True,
        tol=config.DE_TOL
    )

    best_vector = flat_to_formation(result.x, player_names)
    best_cost = result.fun

    return best_vector, best_cost, history
