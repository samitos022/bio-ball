import cma
import numpy as np
from utils.animation import save_generation_plot
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function 
import config

def run_optimization(initial_guess, obstacles, ball_position, player_names):
    """
    Esegue CMA-ES e restituisce sia la formazione ottima che la storia dei costi.
    """
    
    # 1. Preparazione argomenti statici
    initial_df_ref = flat_to_formation(initial_guess, player_names)
    fitness_args = (player_names, obstacles, ball_position, initial_df_ref)
    
    # 2. Configurazione CMA
    sigma_init = config.CMA_SIGMA_INIT
    options = {
        'maxiter': config.CMA_MAXITER,
        'popsize': config.CMA_POPSIZE,
        'bounds': [0, 1], 
        'verbose': -1,
        'tolfun': config.CMA_TOLFUN
    }

    # Inizializziamo la strategia
    es = cma.CMAEvolutionStrategy(initial_guess, sigma_init, options)
    
    cost_history = []

    print(f"Inizio ottimizzazione (Ball: {ball_position})...")

    # 3. Loop di ottimizzazione manuale
    while not es.stop():
        # A. Chiediamo una nuova popolazione di soluzioni candidate
        solutions = es.ask()
        
        # B. Valutiamo tutte le soluzioni
        fitness_values = [objective_function(x, fitness_args) for x in solutions]
        
        # C. Aggiorniamo la strategia con i risultati
        es.tell(solutions, fitness_values)
        
        # D. Salviamo il miglior risultato di questa generazione per il grafico
        current_best = min(fitness_values)
        cost_history.append(current_best)
        
        idx_best = np.argmin(fitness_values)
        best_vector_gen = solutions[idx_best]
        best_cost_gen = fitness_values[idx_best]

        cost_history.append(best_cost_gen)

        save_generation_plot(
            best_vector_gen,
            player_names,
            obstacles,
            ball_position,
            es.countiter
        )

        # Log opzionale in console ogni 10 generazioni
        if es.countiter % 10 == 0:
            print(f"Gen {es.countiter}: Costo {current_best:.4f}")

    # 4. Recuperiamo il risultato finale
    result = es.result
    best_vector = flat_to_formation(result.xbest, player_names)
    best_score = result.fbest
    
    print(f"Ottimizzazione completata. Costo finale: {best_score:.4f}")
    
    return best_vector, cost_history