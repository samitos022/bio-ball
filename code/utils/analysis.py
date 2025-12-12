import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import os

def possessions(match):
    """Parses match events to determine which team had possession for each frame."""
    possessions_dict = {}
    current_team = None
    current_start = None
    current_end = None

    for i, row in match.iterrows():
        team = row["Team"]
        start = row["Start Frame"]
        end = row["End Frame"]

        if pd.isna(team) or pd.isna(start) or pd.isna(end): continue
        if start < 0 or end <= start: continue

        if current_team is None:
            current_team, current_start, current_end = team, start, end
            continue

        if team == current_team:
            current_end = max(current_end, end)
        else:
            for j in range(current_start, current_end + 1):
                possessions_dict[j] = current_team
            current_team, current_start, current_end = team, start, end

    if current_team is not None:
        for f in range(current_start, current_end + 1):
            possessions_dict[f] = current_team

    return possessions_dict

def get_phase(row, team, period):
    """Determines the game phase (Attacking/Defensive possession or Defensive phase)."""
    if pd.isna(row["Ball_x"]) or pd.isna(row["Possession"]): return None

    # Determine phase based on possession and ball location
    if row["Possession"] == team:
        # Team has possession
        if period == 1:
            return "Attacking possession" if ((team == "Home" and row["Ball_x"] > 0.5) or (team == "Away" and row["Ball_x"] < 0.5)) else "Defensive possession"
        elif period == 2:
            return "Attacking possession" if ((team == "Home" and row["Ball_x"] < 0.5) or (team == "Away" and row["Ball_x"] > 0.5)) else "Defensive possession"
    else:
        # Team does not have possession
        return "Defensive phase"

    return None

def average_positions(match, tracking, team):
    """Calculates average player positions for each of the 3 game phases."""
    possessions_dict = possessions(match) 
    tracking['Possession'] = tracking['Frame'].map(possessions_dict)
    
    starters_team = starters(tracking)
    tracking = tracking.copy()
    tracking["Phase"] = tracking.apply(lambda r: get_phase(r, team, r["Period"]), axis=1)

    # Normalize coordinates for 2nd period to match 1st period direction
    mask = tracking["Period"] == 2
    for col in tracking.columns:
        if (col.endswith("_x") or col == "Ball_x") or (col.endswith("_y") or col == "Ball_y"):
            tracking.loc[mask, col] = 1 - tracking.loc[mask, col]

    results = {}
    phases = ["Attacking possession", "Defensive possession", "Defensive phase"]

    for phase in phases:
        phase_df = tracking[tracking["Phase"] == phase]
        if phase_df.empty: continue

        x_cols = [col for col in phase_df.columns if "_x" in col and "Player" in col]
        y_cols = [col for col in phase_df.columns if "_y" in col and "Player" in col]

        avg_positions = pd.DataFrame({
            "x": phase_df[x_cols].mean().values,
            "y": phase_df[y_cols].mean().values
        }, index=[col.replace("_x", "") for col in x_cols])

        if starters_team is not None:
            avg_positions = avg_positions.loc[avg_positions.index.isin(starters_team)]

        results[phase] = avg_positions

    return results

def starters(tracking, n_players=11):
    """Identifies the starting XI based on frame presence."""
    player_cols = [col for col in tracking.columns if '_x' in col]
    player_presence = {
        col.replace('_x', ''): tracking[col].notna().sum() 
        for col in player_cols
    }
    sorted_players = sorted(player_presence.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_players[:n_players]]

def plot_formation(positions, title, team='Home', color='red'):
    """Simple scatter plot of the formation."""
    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')

    if isinstance(positions, dict):
        xs, ys = zip(*positions.values())
        labels = list(positions.keys())
    else:
        xs, ys = positions['x'], positions['y']
        labels = positions.index

    pitch.scatter(xs, ys, ax=ax, s=150, c=color, edgecolors='white', zorder=3)
    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=20)
    plt.show()

def plot_formation_with_ball_and_obstacles(positions, title, team='Home', color='red', ball_position=None, obstacles=None):
    """Plots formation, ball, and opponents on a horizontal pitch."""
    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')

    # Home Team
    if isinstance(positions, dict):
        xs, ys = zip(*positions.values())
        labels = list(positions.keys())
    else:
        xs, ys = positions['x'], positions['y']
        labels = positions.index

    pitch.scatter(xs, ys, ax=ax, s=200, c=color, edgecolors='white', zorder=3, label=team)
    
    # Labels
    for i, label in enumerate(labels):
        pitch.annotate(label, (xs[i], ys[i]), ax=ax, fontsize=8, ha='center', va='center', color='white', zorder=4)

    # Obstacles (Away)
    if obstacles is not None:
        obs_x = obstacles["x"].values if hasattr(obstacles, "columns") else obstacles[:, 0]
        obs_y = obstacles["y"].values if hasattr(obstacles, "columns") else obstacles[:, 1]
        pitch.scatter(obs_x, obs_y, ax=ax, s=200, c='#555555', edgecolors='#888888', alpha=0.7, zorder=2, label='Opponent')

    # Ball
    if ball_position is not None:
        pitch.scatter(ball_position[0], ball_position[1], ax=ax, s=150, c='yellow', edgecolors='black', lw=1.5, zorder=5, label='Ball')
        
    ax.legend(facecolor='#22312b', edgecolor='white', labelcolor='white', loc='upper right')
    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=20)
    plt.show()

def plot_formation_vertical(positions, title, team='Home', color='red', ball_position=None, obstacles=None, save_path=None):
    """Plots formation on a vertical pitch (useful for reports)."""
    pitch = VerticalPitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(8, 12))
    fig.set_facecolor('#22312b')

    # Home Team
    if isinstance(positions, dict):
        xs, ys = zip(*positions.values())
    else:
        xs, ys = positions['x'].values, positions['y'].values

    pitch.scatter(xs, ys, ax=ax, s=200, c=color, edgecolors='white', zorder=3, label=team)

    # Obstacles
    if obstacles is not None:
        obs_x = obstacles["x"].values if hasattr(obstacles, "columns") else obstacles[:, 0]
        obs_y = obstacles["y"].values if hasattr(obstacles, "columns") else obstacles[:, 1]
        pitch.scatter(obs_x, obs_y, ax=ax, c='#555555', alpha=0.7, edgecolors='#888888', s=200, zorder=2, label='Opponents')

    # Ball
    if ball_position is not None:
        pitch.scatter(ball_position[0], ball_position[1], ax=ax, s=150, c='yellow', edgecolors='black', lw=1.5, zorder=5, label='Ball')

    ax.legend(facecolor='#22312b', edgecolor='white', labelcolor='white', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    ax.set_title(f"{team} – {title}", color='white', fontsize=18, pad=40)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()

def plot_convergence(history, save_path=None):
    """Plots the optimization fitness cost over generations."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#22312b')
    plt.gcf().set_facecolor('#22312b')
    
    plt.plot(history, color='#00ff85', linewidth=2.5, label='Best Cost per Generation')
    plt.title('Optimization Convergence', color='white', fontsize=16, pad=15)
    plt.xlabel('Generations', color='white', fontsize=12)
    plt.ylabel('Cost Function (Minimization)', color='white', fontsize=12)
    plt.grid(color='white', alpha=0.1, linestyle='--')
    plt.tick_params(colors='white')
    
    for spine in ax.spines.values(): spine.set_color('white')

    plt.legend(facecolor='#22312b', edgecolor='white', labelcolor='white')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved: {save_path}")

    plt.show()

def prepare_obstacles(avg_positions_dict, phase, starters_list):
    """Extracts opponent positions for the specified phase."""
    if phase not in avg_positions_dict:
        raise ValueError(f"Phase '{phase}' not found in opponent data.")
    
    df_obstacles = avg_positions_dict[phase]
    df_obstacles = df_obstacles.reindex(starters_list).dropna()
    
    if len(df_obstacles) < 11:
        print(f"Warning: Found only {len(df_obstacles)} opponents for phase {phase}.")

    return df_obstacles[['x', 'y']].values