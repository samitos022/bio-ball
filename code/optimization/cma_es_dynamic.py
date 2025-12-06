import cma
import numpy as np
from utils.animation_dynamic import interpolate_vectors, save_generation_plot
from utils.conversion import flat_to_formation
from optimization.objectives import objective_function
from utils.away_reaction import react_away_to_home
import config

# ------------------------------------------------------------------------------
# CMA-ES PRINCIPALE
# ------------------------------------------------------------------------------
def run_optimization(initial_guess, initial_away_df, ball_position, player_names):

    # Argomenti statici per la fitness
    initial_df_ref = flat_to_formation(initial_guess, player_names)
    fitness_args = (player_names, initial_away_df, ball_position, initial_df_ref, 'dynamic')
    
    # Parametri CMA
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

    print("Inizio ottimizzazione...")
    
    prev_best_vector = initial_guess.copy()

    # Ogni quante generazioni salvare un blocco di animazione
    SAVE_EVERY = 2
    INTERPOLATION_STEPS = 6

    # ----------------------------------------------------------------------
    #                        LOOP DI OTTIMIZZAZIONE
    # ----------------------------------------------------------------------
    while not es.stop():

        solutions = es.ask()
        fitness_values = []

        best_vector_home = None
        best_df_home = None
        best_df_away = None
        best_cost = None

        # Valutazione popolazione
        for x in solutions:

            # Calcolo costo
            f = objective_function(x, fitness_args)
            fitness_values.append(f)

            # Formazione home
            df_home = flat_to_formation(x, player_names)

            # Away reattiva
            df_away = react_away_to_home(
                home_df=df_home,
                base_away_df=initial_away_df,
                ball_pos=ball_position
            )

            # Prendi il migliore della generazione
            if best_cost is None or f < best_cost:
                best_cost = f
                best_vector_home = x
                best_df_home = df_home
                best_df_away = df_away

        # Aggiornamento CMA
        es.tell(solutions, fitness_values)
        cost_history.append(best_cost)

        # ------------------------------------------------------------------
        #               SALVATAGGIO FRAME PER ANIMAZIONE (FLUIDA)
        # ------------------------------------------------------------------
        if es.countiter % SAVE_EVERY == 0 and best_vector_home is not None:

            # Interpolazione tra la generazione precedente e quella attuale
            inter_frames = interpolate_vectors(
                prev_best_vector,
                best_vector_home,
                steps=INTERPOLATION_STEPS
            )

            # Salvo frame interpolati
            for f in inter_frames:
                save_generation_plot(
                    f,
                    player_names,
                    best_df_away,
                    ball_position,
                    es.countiter
                )

            # Aggiorna “ultimo best” per la futura interpolazione
            prev_best_vector = best_vector_home.copy()

        # Log ogni 10 iterazioni
        if es.countiter % 10 == 0:
            print(f"Gen {es.countiter}: best cost = {best_cost:.4f}")

    # Fine CMA
    print(f"Completato. Costo finale = {best_cost:.4f}")

    best_vector = es.result.xbest
    best_solution_df = flat_to_formation(best_vector, player_names)

    return best_solution_df, cost_history
