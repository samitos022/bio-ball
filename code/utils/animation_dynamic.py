import os
import imageio
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np
from utils.conversion import flat_to_formation

# cartella dove salvare i frame
FRAMES_DIR = "code/animation_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# RISOLUZIONE FISSA GARANTITA (stessa dimensione ogni volta)
FIGSIZE = (10, 6)
DPI = 100

def interpolate_vectors(vec_prev, vec_next, steps=4):
    """
    Interpolazione lineare leggera per generare frame fluidi.
    vec_prev e vec_next sono vettori flat di posizioni.
    """
    frames = []
    for alpha in np.linspace(0, 1, steps):
        interpolated = vec_prev * (1 - alpha) + vec_next * alpha
        frames.append(interpolated)
    return frames

def save_generation_plot(vector, player_names, df_away, ball_position, gen_id):
    """
    Salva un frame dell'animazione. Sempre stessa risoluzione.
    """
    df_home = flat_to_formation(vector, player_names)

    pitch = Pitch(
        pitch_type="metricasports",
        pitch_length=106,
        pitch_width=68,
        pitch_color="#22312b",
        line_color="white"
    )

    # ⚠️ draw NON accetta dpi, quindi solo figsize
    fig, ax = pitch.draw(figsize=FIGSIZE)
    fig.set_facecolor("#22312b")

    # --- HOME TEAM ---
    xs = df_home["x"].values
    ys = df_home["y"].values
    pitch.scatter(xs, ys, ax=ax, c="blue", s=120, edgecolors="white", zorder=3)

    # --- AWAY TEAM ---
    obs_x = df_away["x"].values
    obs_y = df_away["y"].values
    pitch.scatter(obs_x, obs_y, ax=ax, c="red", s=120, edgecolors="white", zorder=3)

    # --- BALL ---
    pitch.scatter(
        ball_position[0], ball_position[1],
        ax=ax, c="yellow", s=180, edgecolors="black", zorder=4
    )

    ax.set_title(f"Generation {gen_id}", color="white", fontsize=18)

    # Salvo con DPI fisso qui → questo garantisce shape identica per tutti i frame
    frame_path = os.path.join(FRAMES_DIR, f"frame_{gen_id:04d}.png")
    plt.savefig(frame_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def create_evolution_gif(output="evolution.gif", duration=0.15):
    """
    Carica TUTTI i frame e crea la GIF, garantendo shape identici.
    """
    files = sorted(os.listdir(FRAMES_DIR))
    images = []

    print(f"[INFO] Creo GIF con {len(files)} frame...")

    # Carica e verifica shape
    shapes = set()

    for fname in files:
        path = os.path.join(FRAMES_DIR, fname)
        img = imageio.imread(path)
        shapes.add(img.shape)
        images.append(img)

    if len(shapes) > 1:
        print("[ERROR] I frame NON hanno tutti la stessa shape!")
        print(shapes)
        print("→ Questo NON dovrebbe accadere con FIGSIZE e DPI fissati!")
        return

    print("[INFO] Tutti i frame hanno la stessa shape:", shapes)

    # Salva GIF
    imageio.mimsave(output, images, duration=duration)
    print(f"[SUCCESS] GIF salvata come {output}")
