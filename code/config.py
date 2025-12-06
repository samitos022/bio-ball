# config.py

# =============================================================================
# PARAMETRI CAMPO E GIOCO
# =============================================================================
FIELD_LIMITS = (105.0, 68.0)  # Dimensioni campo standard (x, y)
OFFSIDE_ATTACK_DIR = 'right'  # Direzione di attacco per il fuorigioco

# =============================================================================
# PESI DELLA FUNZIONE OBIETTIVO (OBJECTIVE FUNCTION)
# =============================================================================
# Questi sono i pesi che determinano la "tattica". 
# Optuna potrà modificarli per cambiare il comportamento della squadra.
OBJ_W_CONSTRAINTS = 100.0   # Peso per penalità "hard" (fuori campo, sovrapposizioni)
OBJ_W_COVER       = 1.0     # Peso copertura spaziale (Convex Hull)
OBJ_W_PASS        = 1.5     # Peso qualità linee di passaggio
OBJ_W_BALL        = 20.0    # Peso supporto al portatore di palla
OBJ_W_OFFSIDE     = 100.0   # Peso mantenimento linea fuorigioco

# =============================================================================
# PARAMETRI COSTI SPECIFICI
# =============================================================================

# --- Costo Passaggi ---
PASS_BLOCK_THRESHOLD = 0.03  # Distanza minima ostacolo-linea per considerare passaggio bloccato
PASS_W_BLOCK = 8.0           # Penalità se il passaggio è bloccato
PASS_W_LONG = 1.5            # Penalità per passaggi troppo lunghi
PASS_W_ANGLE = 0.5           # Penalità per angoli di passaggio difficili
PASS_MAX_LEN = 0.35          # Lunghezza massima "ideale" (normalizzata 0-1 o in metri/100)
PASS_PENALTY_NO_OPTS = 10.0  # Penalità extra se non ci sono passaggi sicuri

# --- Costo Supporto Palla ---
BALL_SUPPORT_W_MULT = 5.0    # Moltiplicatore distanza supporto

# =============================================================================
# PENALITÀ FISICHE E TATTICHE (CONSTRAINTS)
# =============================================================================
PENALTY_MAX_THRESHOLD = 5000 # Se i vincoli superano questo valore, interrompi calcolo fine
MIN_DIST_PLAYER = 0.02       # Distanza minima tra giocatori (evitare collisioni)

PENALTY_W_TRANSITION = 1.0   # Costo spostamento (per scenario dinamico)
PENALTY_W_BOUNDARY   = 100.0 # Costo uscita dal campo
PENALTY_W_PROXIMITY  = 500.0 # Costo collisione tra giocatori
PENALTY_W_ORDER      = 10.0  # Mantenimento ordine relativo (dx resta a dx di sx)

# =============================================================================
# PARAMETRI SOLVER (OTTIMIZZATORI)
# =============================================================================

# --- Differential Evolution (DE) ---
DE_MAXITER = 50
DE_POPSIZE = 20
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6

# --- CMA-ES ---
CMA_MAXITER = 100
CMA_POPSIZE = 14
CMA_SIGMA_INIT = 0.05
CMA_TOLFUN = 1e-3