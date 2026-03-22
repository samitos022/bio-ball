# config_gecco.py

# --- PARAMETRI SPAZIALI (PITCH CONTROL & xT) ---
GRID_RES = 50           
PLAYER_INFLUENCE_R = 0.15 
EPSILON = 1e-5          

# --- PESI DELLA FUNZIONE OBIETTIVO (NORMALIZZATI) ---
# Scalati per mantenere la fitness entro ordini di grandezza gestibili (es. 0.1 -> 100)
LAMBDA_GHOSTING = 3.5    # (Prima era 3500)

# --- VINCOLI ---
FIELD_LIMITS = (1.0, 1.0)
GOALKEEPER_AREA = {
    "x_min": 0.0, "x_max": 0.165,
    "y_min": 0.21, "y_max": 0.79
}
MIN_DIST_PLAYER = 0.10   # 0.10 = ~10 metri

# --- PESI DELLE PENALITÀ HARD/SOFT (DIVISI PER 1000) ---
PENALTY_W_BOUNDARY   = 10.0    # (Prima 10000)
PENALTY_W_GOALKEEPER = 10.0    # (Prima 10000)
PENALTY_W_PROXIMITY  = 50.0    # (Prima 50000) - Più grave uscire dal campo
PENALTY_W_BALL       = 150.0   # (Prima 150000) - Importante pressare la palla
PENALTY_W_OFFSIDE    = 200.0   # (Prima 200000) - Vincolo supremo (non va MAI violato)

# --- PESI TATTICI ---
# Visto che abbiamo abbassato le penalità, i pesi tattici possono rimanere simili
# o essere leggermente adattati se noti che l'algoritmo ignora le penalità.
WEIGHT_TACTICAL_ATTACK = 20.0   
WEIGHT_REST_DEFENSE    = 20.0    
WEIGHT_TACTICAL_DEFEND = 20.0