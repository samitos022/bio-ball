import numpy as np
from scipy.optimize import differential_evolution
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function
from utils.away_reaction import react_away_to_home
import config

# Aggiunto phase_name
def run_de_optimization(initial_guess, initial_away_df, ball_position, player_names, phase_name="Fase difensiva"):
    dim = len(initial_guess)
    bounds = [(0, 1)] * dim
    history = []
    step_counter = 0
    
    initial_df_ref = flat_to_formation(initial_guess, player_names)

    # Wrapper per DE che chiama la nostra objective_function
    def fitness_wrapper(x):
        # Passiamo mode='dynamic' perché DE usa react_away_to_home internamente se necessario,
        # ma nel tuo codice originale la reazione era gestita qui nel wrapper.
        # Per coerenza con objectives.py, facciamo gestire tutto a objective_function?
        # NO: DE di scipy richiede una funzione semplice.
        # Manteniamo la logica 'dynamic' passando gli argomenti corretti.
        
        args = (player_names, initial_away_df, ball_position, initial_df_ref, phase_name, 'dynamic')
        return objective_function(x, args)

    print(f"Inizio ottimizzazione DE ({phase_name})...")

    def callbackF(xk, convergence):
        nonlocal step_counter
        step_counter += 1
        cost = fitness_wrapper(xk)
        history.append(cost)
        if step_counter % 10 == 0:
            print(f"DE Step {step_counter}: f(x)={cost:.4f}")

    result = differential_evolution(
        func=fitness_wrapper,
        bounds=bounds,
        maxiter=config.DE_MAXITER,
        popsize=config.DE_POPSIZE,
        mutation=config.DE_MUTATION,
        recombination=config.DE_RECOMBINATION,
        callback=callbackF,
        polish=True,
        tol=config.DE_TOL
    )

    return result.x, result.fun, history