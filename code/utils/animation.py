import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os
from utils.conversion import flat_to_formation
import imageio

def save_generation_plot(vector, player_names, obstacles, ball_pos, gen, output_dir="code/animation_frames"):
    # Crea cartella
    os.makedirs(output_dir, exist_ok=True)

    # Converti vettore → dataframe
    df = flat_to_formation(vector, player_names)

    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='white')

    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor('#22312b')

    # Plot giocatori
    pitch.scatter(df['x'], df['y'], ax=ax, c='blue', s=150, zorder=3, edgecolors='white')

    # Plot ostacoli (away)
    if obstacles is not None:
        pitch.scatter(obstacles[:,0], obstacles[:,1], ax=ax, c='red', s=80, zorder=3)

    # Plot palla
    pitch.scatter(ball_pos[0], ball_pos[1], ax=ax, c='yellow', s=200, zorder=5)

    # Titolo
    ax.set_title(f"Generation {gen}", color='white', fontsize=16)

    # Salva immagine
    filepath = os.path.join(output_dir, f"gen_{gen:04d}.png")
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def create_evolution_gif(frame_dir="code/animation_frames", output="animation.gif"):
    frames = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")]
    )

    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(output, images, duration=0.15)

    print(f"GIF creata: {output}")