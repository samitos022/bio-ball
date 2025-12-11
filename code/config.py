# config.py

# PARAMETRI GENERALI
FIELD_LIMITS = (1.0, 1.0)
MIN_DIST_PLAYER = 0.02
OFFSIDE_ATTACK_DIR = 'right'

# PARAMETRI AREA PORTIERE
GOALKEEPER_AREA = {
    "x_min": 0.0,
    "x_max": 0.165,
    "y_min": 0.21,
    "y_max": 0.79
}

# PARAMETRI CONSTRAINTS (Sempre attivi)
PENALTY_MAX_THRESHOLD = 5000
OBJ_W_CONSTRAINTS = 50.0
PENALTY_W_BOUNDARY   = 100.0
PENALTY_W_GOALKEEPER = 5000
PENALTY_W_PROXIMITY  = 50.0
PENALTY_W_TRANSITION = 0.1
PENALTY_W_ORDER      = 5.0

# PARAMETRI FISICI COSTI
PASS_MAX_LEN = 0.45
PASS_BLOCK_THRESHOLD = 0.03
PASS_W_LONG = 1.5
PASS_W_ANGLE = 0.5
PASS_W_BLOCK = 1.0
PASS_PENALTY_NO_OPTS = 10.0

# NUOVI PARAMETRI SATURAZIONE
# Punteggio di qualità cumulativo target. 
# Una volta raggiunto questo score, il costo scende drasticamente verso zero.
PASS_TARGET_SCORE_OFF = 2.5  # Attacco: bastano ~2 passaggi di alta qualità (verticali)
PASS_TARGET_SCORE_DEF = 3.5  # Difesa: servono ~3-4 passaggi sicuri (anche laterali)

# === CONFIGURAZIONE PESI PER FASE ===
# Se un peso è 0.0, l'obiettivo non viene calcolato per quella fase.

PHASE_WEIGHTS = {
    "Fase difensiva": {
        # Obiettivi Difensivi
        "W_MARKING":      45.0,   # Priorità: marcare
        "W_COMPACTNESS":  5.0,   # Priorità: stare stretti
        "W_LINE_HEIGHT":  4.0,    # Priorità: tenere la linea alta
        "W_BALL_PRESS":   20.0,   # Priorità: pressare portatore
        
        # Obiettivi Offensivi (Disattivati o irrilevanti)
        "W_COVERAGE":     0.0,    # Non vogliamo allargarci a caso
        "W_PASSING":      0.0,    # Non abbiamo la palla
        "W_OFFSIDE":      0.0,     # Noi non andiamo in fuorigioco
        "W_PREV_MARKING": 0.0
    },
    
    "Possesso offensivo": {
        "W_MARKING": 2.6146,
        "W_COVERAGE": 34.7123,
        #"W_PASSING": 10.2512,
        #"W_OFFSIDE": 38.8458,
        "W_BALL_PRESS": 30.0,
        

        # Obiettivi Difensivi (Disattivati)
        #"W_MARKING":      0.0,
        "W_COMPACTNESS":  0.0,
        "W_LINE_HEIGHT":  0.0,
        
        # Obiettivi Offensivi
        #"W_COVERAGE":     20.0,    # Allargare il campo
        "W_PASSING":      6.0,    # Trovare linee
        "W_OFFSIDE":      100.0,    # Evitare fuorigioco (Regola)
        "W_PREV_MARKING": 15.0
    },
    
    "Possesso difensivo": {
        # Fase di Costruzione
        "W_OFFSIDE":      100.0,
        "W_COVERAGE": 23.6387,
        "W_PASSING": 35.3722,
        "W_BALL_PRESS": 10.0101,
        "W_MARKING": 61.0412,
        "W_COMPACTNESS": 87.8870,
        "W_LINE_HEIGHT": 29.8935,
        "W_PREV_MARKING": 14.2057
    }
}

# PARAMETRI SOLVER
CMA_MAXITER = 200
CMA_POPSIZE = 20
CMA_SIGMA_INIT = 0.15
CMA_TOLFUN = 1e-4

DE_MAXITER = 50
DE_POPSIZE = 20
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6