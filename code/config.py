# config.py

# PARAMETRI GENERALI
FIELD_LIMITS = (1.0, 1.0)
MIN_DIST_PLAYER = 0.02
OFFSIDE_ATTACK_DIR = 'right'

# PARAMETRI CONSTRAINTS (Sempre attivi)
PENALTY_MAX_THRESHOLD = 5000
OBJ_W_CONSTRAINTS = 100.0
PENALTY_W_BOUNDARY   = 100.0
PENALTY_W_PROXIMITY  = 500.0
PENALTY_W_TRANSITION = 1.0
PENALTY_W_ORDER      = 10.0

# PARAMETRI FISICI COSTI
PASS_MAX_LEN = 0.45
PASS_BLOCK_THRESHOLD = 0.03
PASS_W_LONG = 1.5
PASS_W_ANGLE = 0.5
PASS_W_BLOCK = 10.0
PASS_PENALTY_NO_OPTS = 10.0

# === CONFIGURAZIONE PESI PER FASE ===
# Se un peso è 0.0, l'obiettivo non viene calcolato per quella fase.

PHASE_WEIGHTS = {
    "Fase difensiva": {
        # Obiettivi Difensivi
        "W_MARKING":      40.0,   # Priorità: marcare
        "W_COMPACTNESS":  5.0,   # Priorità: stare stretti
        "W_LINE_HEIGHT":  5.0,    # Priorità: tenere la linea alta
        "W_BALL_PRESS":   20.0,   # Priorità: pressare portatore
        
        # Obiettivi Offensivi (Disattivati o irrilevanti)
        "W_COVERAGE":     0.0,    # Non vogliamo allargarci a caso
        "W_PASSING":      0.0,    # Non abbiamo la palla
        "W_OFFSIDE":      0.0,     # Noi non andiamo in fuorigioco
        "W_PREV_MARKING": 0.0
    },
    
    "Possesso offensivo": {
        # Obiettivi Difensivi (Disattivati)
        "W_MARKING":      0.0,
        "W_COMPACTNESS":  0.0,
        "W_LINE_HEIGHT":  0.0,
        "W_BALL_PRESS":   5.0,    # Ball Support (basso, non affollare)
        
        # Obiettivi Offensivi
        "W_COVERAGE":     20.0,    # Allargare il campo
        "W_PASSING":      6.0,    # Trovare linee
        "W_OFFSIDE":      50.0,    # Evitare fuorigioco (Regola)
        "W_PREV_MARKING": 15.0
    },
    
    "Possesso difensivo": {
        # Fase di Costruzione
        "W_MARKING":      0.0,
        "W_COMPACTNESS":  2.0,    # Un po' compatti per sicurezza
        "W_LINE_HEIGHT":  0.0,
        "W_BALL_PRESS":   15.0,   # Ball Support (Alto, servono appoggi)
        
        "W_COVERAGE":     2.0,    # Copertura media
        "W_PASSING":      15.0,   # Passaggi sicuri priorità assoluta
        "W_OFFSIDE":      0.0,     # Difficile essere in offside in difesa
        "PREV_MARKING":   5.0 
    }
}

# PARAMETRI SOLVER
CMA_MAXITER = 100
CMA_POPSIZE = 20
CMA_SIGMA_INIT = 0.05
CMA_TOLFUN = 1e-4

DE_MAXITER = 50
DE_POPSIZE = 20
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6