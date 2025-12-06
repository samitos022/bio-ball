import matplotlib.pyplot as plt
from matplotlib import animation
from mplsoccer import Pitch
import numpy as np
from utils.load_data import load_and_clean_metrica_tracking
from utils.load_data import load_match

try:
    tracking_home = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Home_Team.csv')
    tracking_away = load_and_clean_metrica_tracking('data/metrica/sample_game_1/Sample_Game_1_RawTrackingData_Away_Team.csv')
    print("[SUCCESS] Dati caricati e puliti.")
    game = load_match('data/metrica/sample_game_1/Sample_Game_1_RawEventsData.csv')
except Exception as e:
    print(f"[ERROR] Impossibile caricare i dati: {e}")
    exit()


pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))
fig.set_facecolor('#22312b')

# Frame 1
initial_frame = tracking_home.loc[tracking_home['Frame'] == 1]
home_x = initial_frame.filter(like='Player').filter(like='_x').iloc[0].dropna()
home_y = initial_frame.filter(like='Player').filter(like='_y').iloc[0].dropna()
away_x = initial_frame.filter(like='Player').filter(like='_x').iloc[0].dropna()
away_y = initial_frame.filter(like='Player').filter(like='_y').iloc[0].dropna()
ball_pos = initial_frame[['Ball_x', 'Ball_y']].iloc[0]

home_scatter = pitch.scatter(home_x, home_y, ax=ax, s=300, c='red', zorder=3)
away_scatter = pitch.scatter(away_x, away_y, ax=ax, s=300, c='blue', zorder=3)
ball_scatter = pitch.scatter(ball_pos['Ball_x'], ball_pos['Ball_y'], ax=ax, s=150, c='yellow', zorder=5)

title = ax.set_title("Frame 0", color='white', fontsize=16)

def update(frame_num):
    """Questa funzione viene chiamata per ogni frame dell'animazione."""
    current_frame = frame_num + 1 
    
    frame_home = tracking_home.loc[tracking_home['Frame'] == current_frame]
    frame_away = tracking_away.loc[tracking_away['Frame'] == current_frame]
    
    home_x = frame_home.filter(like='Player').filter(like='_x').iloc[0].dropna()
    home_y = frame_home.filter(like='Player').filter(like='_y').iloc[0].dropna()
    away_x = frame_away.filter(like='Player').filter(like='_x').iloc[0].dropna()
    away_y = frame_away.filter(like='Player').filter(like='_y').iloc[0].dropna()
    ball_pos = frame_home[['Ball_x', 'Ball_y']].iloc[0]
    
    home_scatter.set_offsets(np.c_[home_x, home_y])
    away_scatter.set_offsets(np.c_[away_x, away_y])
    if not ball_pos.isnull().any():
        ball_scatter.set_offsets(np.c_[ball_pos['Ball_x'], ball_pos['Ball_y']])
        
    title.set_text(f"Frame {current_frame}")

    return home_scatter, away_scatter, ball_scatter, title

num_frames = int(tracking_home['Frame'].max())

anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)

plt.show()