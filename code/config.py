# config.py

# GENERAL PARAMETERS
FIELD_LIMITS = (1.0, 1.0)
MIN_DIST_PLAYER = 0.02
OFFSIDE_ATTACK_DIR = 'right'

# GOALKEEPER AREA PARAMETERS
GOALKEEPER_AREA = {
    "x_min": 0.0,
    "x_max": 0.165,
    "y_min": 0.21,
    "y_max": 0.79
}

# CONSTRAINTS PARAMETERS
PENALTY_MAX_THRESHOLD = 5000
OBJ_W_CONSTRAINTS = 50.0
PENALTY_W_BOUNDARY   = 100.0
PENALTY_W_GOALKEEPER = 5000
PENALTY_W_PROXIMITY  = 50.0
PENALTY_W_TRANSITION = 0.1
PENALTY_W_ORDER      = 5.0

# PHYSICS COSTS PARAMETERS
PASS_MAX_LEN = 0.45
PASS_BLOCK_THRESHOLD = 0.03
PASS_W_LONG = 1.5
PASS_W_ANGLE = 0.5
PASS_W_BLOCK = 1.0
PASS_PENALTY_NO_OPTS = 10.0

# PASSING OPTIONS PARAMETERS
PASS_TARGET_SCORE_OFF = 2.5
PASS_TARGET_SCORE_DEF = 3.5


# === WHEIGHTS CONFIGURATION FOR EACH PHASE ===
# When set to 0.0, that objective doesn't contribute

PHASE_WEIGHTS = {
    "Defensive phase": {
        # Obiettivi Difensivi
        "W_MARKING":      45.0,
        "W_COMPACTNESS":  5.0,
        "W_LINE_HEIGHT":  4.0,
        "W_BALL_PRESS":   20.0,
        
        "W_COVERAGE":     0.0,
        "W_PASSING":      0.0,
        "W_OFFSIDE":      0.0,
        "W_PREV_MARKING": 0.0
    },
    
    "Attacking possession": {
        "W_PASSING":        4.0, 
        "W_OFFSIDE":        100.0,
        "W_PREV_MARKING":   15.0,
        "W_COVERAGE":       34.0,
        "W_BALL_PRESS":     50.0,
        
        # Defensive Objectives
        "W_MARKING":        0.0,
        "W_COMPACTNESS":    0.0,
        "W_LINE_HEIGHT":    2.0,
    },
    
    "Defensive possession": {
        "W_OFFSIDE": 100.0,
        "W_COVERAGE": 32.6387,
        "W_PASSING": 7.3722,
        "W_BALL_PRESS": 30.0101,
        "W_MARKING": 2.50412,
        "W_COMPACTNESS": 6.8870,
        "W_LINE_HEIGHT": 5,
        "W_PREV_MARKING": 14.2057
    }
}

# PARAMETRI SOLVER
CMA_MAXITER = 160
CMA_POPSIZE = 20
CMA_SIGMA_INIT = 0.15
CMA_TOLFUN = 1e-4

DE_MAXITER = 160
DE_POPSIZE = 20
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6