# config.py

# CONFIGURAZIONE CAMPO E FISICA
FIELD_LIMITS = (1.0, 1.0)  # Dimensioni campo normalizzate (x, y)
MIN_DIST_PLAYER = 0.08     # Distanza minima tra giocatori (collisione)

# PESI PENALITÀ (constraints.py / penalty_total)
PENALTY_W_TRANSITION = 1.0     # Quanto pesa spostarsi dalla posizione precedente
PENALTY_W_BOUNDARY   = 1000.0  # Penalità per chi esce dal campo
PENALTY_W_PROXIMITY  = 500.0   # Penalità per collisione tra giocatori
PENALTY_W_ORDER      = 10.0    # Mantenimento ordine relativo (difensori dietro attaccanti)

PENALTY_MAX_THRESHOLD = 5000   # Se i vincoli superano questo valore, l'ottimizzazione scarta la soluzione

# PARAMETRI COSTO PASSAGGI (cost_passing_lanes)
PASS_BLOCK_THRESHOLD = 0.03    # Distanza avversario dalla linea di passaggio per considerarlo bloccato
PASS_MAX_LEN         = 0.35    # Distanza massima passaggio (oltre questa scatta penalità)

PASS_W_BLOCK         = 8.0     # Peso penalità passaggio intercettato
PASS_W_LONG          = 1.5     # Peso penalità passaggio troppo lungo
PASS_W_ANGLE         = 0.5     # Peso penalità angolo difficile
PASS_PENALTY_NO_OPTS = 10.0    # Penalità extra se non esistono passaggi validi

# PARAMETRI SUPPORTO E OFFSIDE
BALL_SUPPORT_W_MULT  = 5.0     # Moltiplicatore distanza giocatore più vicino alla palla
OFFSIDE_ATTACK_DIR   = 'right' # Direzione attacco ('right' o 'left')

# PESI FUNZIONE OBIETTIVO (objective_function)
# Questi pesi bilanciano le varie componenti del costo totale
OBJ_W_CONSTRAINTS = 1.0    # Peso generale dei vincoli strutturali
OBJ_W_COVER       = 1.0    # Peso copertura spaziale (Convex Hull)
OBJ_W_PASS        = 1.0    # Peso qualità linee di passaggio
OBJ_W_BALL        = 20.0   # Peso vicinanza alla palla (supporto)
OBJ_W_OFFSIDE     = 100.0  # Peso regola fuorigioco