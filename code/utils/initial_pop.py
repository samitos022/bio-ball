import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.pyplot as plt

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
                return "Attacking possession"
            elif row["Possession"] == team and row["Ball_x"] <= 0.5:
                return "Defensive possession"
            else:
                return "Defensive phase"
        elif team == "Away":
            if row["Possession"] == team and row["Ball_x"] < 0.5:
                return "Attacking possession"
            elif row["Possession"] == team and row["Ball_x"] >= 0.5:
                return "Defensive possession"
            else:
                return "Defensive phase"

    elif period == 2:
        if team == "Home":
            if row["Possession"] == team and row["Ball_x"] < 0.5:
                return "Attacking possession"
            elif row["Possession"] == team and row["Ball_x"] >= 0.5:
                return "Defensive possession"
            else:
                return "Defensive phase"
        elif team == "Away":
            if row["Possession"] == team and row["Ball_x"] > 0.5:
                return "Attacking possession"
            elif row["Possession"] == team and row["Ball_x"] <= 0.5:
                return "Defensive possession"
            else:
                return "Defensive phase"

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
    phases = ["Attacking possession", "Defensive possession", "Defensive phase"]

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

def average_ball_positions(tracking, team):
    tracking["Phase"] = tracking.apply(lambda r: get_phase(r, team, r["Period"]), axis=1)

    results = {}

    for phase in ["Attacking possession", "Defensive possession", "Defensive phase"]:
        phase_df = tracking[tracking["Phase"] == phase]

        if phase_df.empty:
            continue

        ball_x = phase_df["Ball_x"].mean()
        ball_y = phase_df["Ball_y"].mean()

        results[phase] = np.array([ball_x, ball_y])

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
