import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from utils.load_data import load_and_clean_metrica_tracking
from utils.load_data import load_match
from optimization.constraints import penalty_total

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

try:
    tracking_home = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv')
    print("[SUCCESS] Dati caricati e puliti.")
    match = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')
    print("[SUCCESS] Partita caricata.")
except Exception as e:
    print(f"[ERROR] Impossibile caricare i dati: {e}")
    exit()

initial_pop_home = average_positions(match, tracking_home, 'Home')
initial_pop_away = average_positions(match, tracking_away, 'Away')

print(penalty_total(initial_pop_home))

plot_formation(initial_pop_home.get('Possesso offensivo'), 'Possesso offensivo', 'Home')
plot_formation(initial_pop_away.get('Possesso offensivo'), 'Possesso offensivo', 'Away')

plot_formation(initial_pop_home.get('Possesso difensivo'), 'Possesso difensivo', 'Home')
plot_formation(initial_pop_away.get('Possesso difensivo'), 'Possesso difensivo', 'Away')

plot_formation(initial_pop_home.get('Fase difensiva'), 'Fase difensiva', 'Home')
plot_formation(initial_pop_away.get('Fase difensiva'), 'Fase difensiva', 'Away')