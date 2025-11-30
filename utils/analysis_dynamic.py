import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import os

def possessions(match):
    possessions_dict = {}
    current_team = None
    current_start = None
    current_end = None

    for i, row in match.iterrows():
        team = row["Team"]
        start = row["Start Frame"]
        end = row["End Frame"]

        if pd.isna(team) or pd.isna(start) or pd.isna(end):
            continue
        
        if start < 0 or end <= start:
            continue

        if current_team is None:
            current_team = team
            current_start = start
            current_end = end
            continue

        if team == current_team:
            current_end = max(current_end, end)
        else:
            for j in range(current_start, current_end + 1):
                possessions_dict[j] = current_team

            current_team = team
            current_start = start
            current_end = end

    if current_team is not None:
        for f in range(current_start, current_end + 1):
            possessions_dict[f] = current_team

    return possessions_dict

def get_phase(row, team, period):
    if pd.isna(row["Ball_x"]) or pd.isna(row["Possession"]):
        return None

    if period == 1:
        if team == "Home":
            if row["Possession"] == team and row["Ball_x"] > 0.5:
                return "Possesso offensivo"
            elif row["Possession"] == team and row["Ball_x"] <= 0.5:
                return "Possesso difensivo"
            else:
                return "Fase difensiva"
        elif team == "Away":
            if row["Possession"] == team and row["Ball_x"] < 0.5:
                return "Possesso offensivo"
            elif row["Possession"] == team and row["Ball_x"] >= 0.5:
                return "Possesso difensivo"
            else:
                return "Fase difensiva"

    elif period == 2:
        if team == "Home":
            if row["Possession"] == team and row["Ball_x"] < 0.5:
                return "Possesso offensivo"
            elif row["Possession"] == team and row["Ball_x"] >= 0.5:
                return "Possesso difensivo"
            else:
                return "Fase difensiva"
        elif team == "Away":
            if row["Possession"] == team and row["Ball_x"] > 0.5:
                return "Possesso offensivo"
            elif row["Possession"] == team and row["Ball_x"] <= 0.5:
                return "Possesso difensivo"
            else:
                return "Fase difensiva"

    return None

def average_positions(match, tracking, team):
    possessions_dict = possessions(match) 

    tracking['Possession'] = tracking['Frame'].map(possessions_dict)

    starters_team = starters(tracking)

    tracking = tracking.copy()

    tracking["Phase"] = tracking.apply(lambda r: get_phase(r, team, r["Period"]), axis=1)

    if team == "Home":
        mask = tracking["Period"] == 2
        for col in tracking.columns:
            if col.endswith("_x") or col == "Ball_x":
                tracking.loc[mask, col] = 1 - tracking.loc[mask, col]
            if col.endswith("_y") or col == "Ball_y":
                tracking.loc[mask, col] = 1 - tracking.loc[mask, col]

    elif team == "Away":
        mask = tracking["Period"] == 2
        for col in tracking.columns:
            if col.endswith("_x") or col == "Ball_x":
                tracking.loc[mask, col] = 1 - tracking.loc[mask, col]
            if col.endswith("_y") or col == "Ball_y":
                tracking.loc[mask, col] = 1 - tracking.loc[mask, col]

    results = {}
    phases = ["Possesso offensivo", "Possesso difensivo", "Fase difensiva"]

    for phase in phases:
        phase_df = tracking[tracking["Phase"] == phase]
        if phase_df.empty:
            continue

        x_cols = [col for col in phase_df.columns if "_x" in col and "Player" in col]
        y_cols = [col for col in phase_df.columns if "_y" in col and "Player" in col]

        mean_x = phase_df[x_cols].mean()
        mean_y = phase_df[y_cols].mean()

        avg_positions = pd.DataFrame({
            "x": mean_x.values,
            "y": mean_y.values
        }, index=[col.replace("_x", "") for col in x_cols])

        if starters_team is not None:
            avg_positions = avg_positions.loc[avg_positions.index.isin(starters_team)]

        results[phase] = avg_positions

    return results

def starters(tracking, n_players=11):
    player_cols = [col for col in tracking.columns if '_x' in col]

    player_presence = {}

    for col in player_cols:
        player_name = col.replace('_x', '')
        valid_frames = tracking[col].notna().sum()
        player_presence[player_name] = valid_frames

    sorted_players = sorted(player_presence.items(), key=lambda x: x[1], reverse=True)

    starters = [player for player, _ in sorted_players[:n_players]]
    
    return starters

def plot_formation(positions, title, team='Home', color='red'):
    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='white')

    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')

    if isinstance(positions, dict):
        xs = [v[0] for v in positions.values()]
        ys = [v[1] for v in positions.values()]
        labels = list(positions.keys())
    else:
        xs = positions['x']
        ys = positions['y']
        labels = positions.index

    pitch.scatter(xs, ys, ax=ax, s=150, c=color, edgecolors='white', zorder=3)

    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=20)
    plt.show()

def plot_formation_with_ball_and_obstacles(positions, title, team='Home', color='red', ball_position=None, obstacles=None):
    """
    Visualizza la formazione sul campo, inclusa la palla e opzionalmente gli avversari.
    
    Args:
        positions (pd.DataFrame): DataFrame con colonne 'x', 'y' (Squadra ottimizzata).
        title (str): Titolo del grafico.
        team (str): Nome del team.
        color (str): Colore dei giocatori.
        ball_position (tuple, optional): Coordinate (x, y) della palla.
        obstacles (np.array, optional): Array (N, 2) con le posizioni degli avversari.
    """
    # Setup del campo (stile scuro Metrica)
    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, 
                  pitch_color='#22312b', line_color='#c7d5cc')

    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')

    # 1. Plot della Squadra Ottimizzata (Home)
    if isinstance(positions, dict):
        xs = [v[0] for v in positions.values()]
        ys = [v[1] for v in positions.values()]
        labels = list(positions.keys())
    else:
        xs = positions['x']
        ys = positions['y']
        labels = positions.index

    # Disegna i giocatori
    pitch.scatter(xs, ys, ax=ax, s=200, c=color, edgecolors='white', zorder=3, label=team)
    
    # Aggiungi i numeri o nomi (opzionale, per chiarezza)
    for i, label in enumerate(labels):
        pitch.annotate(label, (xs[i], ys[i]), ax=ax, fontsize=8, ha='center', va='center', color='white', zorder=4)

    # 2. Plot degli Ostacoli (Away/Avversari) - Colore Grigio/Spento
    if hasattr(obstacles, "columns"):
        obs_x = obstacles["x"].values
        obs_y = obstacles["y"].values
    else:
        obs_x = obstacles[:, 0]
        obs_y = obstacles[:, 1]
        
    pitch.scatter(obs_x, obs_y, ax=ax, c='#555555', alpha = 0.7, edgecolors = '#888888', s=200, zorder=2, label='Opponents')

    # 3. Plot della Palla - Colore Giallo brillante
    if ball_position is not None:
        pitch.scatter(ball_position[0], ball_position[1], ax=ax, s=150, c='yellow', edgecolors='black', lw=1.5, zorder=5, label='Ball')
        # Linea tratteggiata dalla palla al giocatore più vicino (opzionale estetica)
        # pitch.lines(ball_position[0], ball_position[1], xs[0], ys[0], ax=ax, color='yellow', alpha=0.3, linestyle='--')

    # Legenda e Titoli
    ax.legend(facecolor='#22312b', edgecolor='white', labelcolor='white', loc='upper right')
    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=20)
    
    plt.show()

def plot_formation_vertical(positions, title, team='Home', color='red', ball_position=None, obstacles=None, save_path=None):
    """
    Visualizza la formazione sul campo VERTICALE senza nomi/numeri.
    """
    # 1. Setup campo verticale
    pitch = VerticalPitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, 
                          pitch_color='#22312b', line_color='#c7d5cc')

    fig, ax = pitch.draw(figsize=(8, 12))
    fig.set_facecolor('#22312b')

    # 2. Preparazione Dati
    if isinstance(positions, dict):
        xs = [v[0] for v in positions.values()]
        ys = [v[1] for v in positions.values()]
    else:
        # Se è un DataFrame
        xs = positions['x'].values
        ys = positions['y'].values

    # 3. Disegna i giocatori (Solo i pallini)
    pitch.scatter(xs, ys, ax=ax, s=200, c=color, edgecolors='white', zorder=3, label=team)
    
    # --- RIMOSSO IL BLOCCO DI CODICE PER LE ETICHETTE (pitch.annotate) ---

    # 4. Plot degli Ostacoli (Avversari)
    if obstacles is not None:
        if hasattr(obstacles, "columns"):
            obs_x = obstacles["x"].values
            obs_y = obstacles["y"].values
        else:
            obs_x = obstacles[:, 0]
            obs_y = obstacles[:, 1]
            
        pitch.scatter(obs_x, obs_y, ax=ax, c='#555555', alpha=0.7, edgecolors='#888888', s=200, zorder=2, label='Opponents')

    # 5. Plot della Palla
    if ball_position is not None:
        pitch.scatter(ball_position[0], ball_position[1], ax=ax, s=150, c='yellow', edgecolors='black', lw=1.5, zorder=5, label='Ball')

    # Legenda e Titoli
    ax.legend(facecolor='#22312b', edgecolor='white', labelcolor='white', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=40)

    if save_path:
        # Crea la cartella se non esiste
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Salva ad alta risoluzione (dpi=300) e ritaglia i bordi bianchi (bbox_inches='tight')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    
    plt.show()

def plot_convergence(history, save_path=None):
    """
    Plotta l'andamento del costo (Fitness) durante le generazioni.
    """
    plt.figure(figsize=(10, 6))
    
    # Stile scuro
    ax = plt.gca()
    ax.set_facecolor('#22312b')
    plt.gcf().set_facecolor('#22312b')
    
    # Plot della curva
    plt.plot(history, color='#00ff85', linewidth=2.5, label='Best Cost per Generation')
    
    # Etichette
    plt.title('Ottimizzazione CMA-ES: Convergenza', color='white', fontsize=16, pad=15)
    plt.xlabel('Generazioni', color='white', fontsize=12)
    plt.ylabel('Funzione di Costo (Minimizzazione)', color='white', fontsize=12)
    
    # Griglia e assi
    plt.grid(color='white', alpha=0.1, linestyle='--')
    plt.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.legend(facecolor='#22312b', edgecolor='white', labelcolor='white')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Grafico salvato in: {save_path}")

    plt.show()

def prepare_obstacles(avg_positions_dict, phase, starters_list):
    """
    Estrae le posizioni degli ostacoli (avversari) per la fase specificata.
    
    Args:
        avg_positions_dict (dict): Dizionario delle posizioni medie (output di average_positions).
        phase (str): La fase in cui si trova l'avversario (es. "Fase difensiva").
        starters_list (list): Lista dei nomi dei titolari per garantire l'ordine.
        
    Returns:
        np.array: Matrice (11, 2) con le coordinate [x, y] degli avversari.
    """
    # 1. Recupera il DataFrame della fase corretta
    # Esempio: avg_positions_dict["Fase difensiva"]
    if phase not in avg_positions_dict:
        raise ValueError(f"Fase '{phase}' non trovata nei dati avversari.")
    
    df_obstacles = avg_positions_dict[phase]
    
    # 2. Filtra e ordina in base alla lista dei titolari
    # Questo assicura che abbiamo esattamente 11 punti e sappiamo chi sono (opzionale, ma pulito)
    # Se un giocatore manca nei dati di quella fase, pandas metterà NaN -> bisogna gestirlo
    df_obstacles = df_obstacles.reindex(starters_list).dropna()
    
    # Check di sicurezza: se mancano dati, avvisiamo
    if len(df_obstacles) < 11:
        print(f"Attenzione: Trovati solo {len(df_obstacles)} ostacoli su 11 richiesti nella fase {phase}.")

    # 3. Conversione in Numpy Array per velocità
    # Output shape: (N_players, 2)
    obstacles_matrix = df_obstacles[['x', 'y']].values
    
    return obstacles_matrix