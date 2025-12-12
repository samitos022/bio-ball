import os
import shutil
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from utils.conversion import flat_to_formation

# Directory to save temporary frames
FRAMES_DIR = "code/animation_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# Fixed resolution configuration to ensure GIF stability
FIGSIZE = (10, 6)
DPI = 100

def interpolate_vectors(vec_prev, vec_next, steps=4):
    """
    Performs linear interpolation between two optimization vectors 
    to generate intermediate frames for a smooth animation.
    """
    frames = []
    for alpha in np.linspace(0, 1, steps):
        interpolated = vec_prev * (1 - alpha) + vec_next * alpha
        frames.append(interpolated)
    return frames

def save_generation_plot(vector, player_names, df_away, ball_position, gen_id):
    """
    Generates and saves a snapshot of the current dynamic generation.
    """
    df_home = flat_to_formation(vector, player_names)

    pitch = Pitch(
        pitch_type="metricasports",
        pitch_length=106,
        pitch_width=68,
        pitch_color="#22312b",
        line_color="white"
    )

    # Note: .draw() accepts figsize but not DPI directly
    fig, ax = pitch.draw(figsize=FIGSIZE)
    fig.set_facecolor("#22312b")

    # --- HOME TEAM ---
    xs = df_home["x"].values
    ys = df_home["y"].values
    pitch.scatter(xs, ys, ax=ax, c="blue", s=120, edgecolors="white", zorder=3)

    # --- AWAY TEAM (Dynamic) ---
    obs_x = df_away["x"].values
    obs_y = df_away["y"].values
    pitch.scatter(obs_x, obs_y, ax=ax, c="red", s=120, edgecolors="white", zorder=3)

    # --- BALL ---
    pitch.scatter(
        ball_position[0], ball_position[1],
        ax=ax, c="yellow", s=180, edgecolors="black", zorder=4
    )

    ax.set_title(f"Generation {gen_id}", color="white", fontsize=18)

    # Save frame
    frame_path = os.path.join(FRAMES_DIR, f"frame_{gen_id:04d}.png")
    plt.savefig(frame_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def create_evolution_gif(output="evolution.gif", duration=0.15):
    """
    Compiles all saved frames into a GIF, ensuring image dimensions are consistent.
    """
    if not os.path.exists(FRAMES_DIR):
        print("Frames directory not found.")
        return

    files = sorted(os.listdir(FRAMES_DIR))
    images = []

    print(f"[INFO] Creating GIF with {len(files)} frames...")

    # Load images and validate shapes
    shapes = set()

    for fname in files:
        path = os.path.join(FRAMES_DIR, fname)
        img = imageio.imread(path)
        shapes.add(img.shape)
        images.append(img)

    if len(shapes) > 1:
        print("[ERROR] Frames do not have consistent shapes!")
        print(f"Shapes found: {shapes}")
        print("→ This typically happens if bbox_inches='tight' alters the size dynamically.")
        return

    # Save GIF
    imageio.mimsave(output, images, duration=duration, loop=0)
    print(f"[SUCCESS] GIF saved as {output}")

    # Cleanup
    shutil.rmtree(FRAMES_DIR)
    print(f"[INFO] Cleanup complete: Removed '{FRAMES_DIR}'")