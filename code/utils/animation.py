import os
import shutil
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mplsoccer import Pitch
from utils.conversion import flat_to_formation

def save_generation_plot(vector, player_names, obstacles, ball_pos, gen, output_dir="code/animation_frames"):
    """
    Generates and saves a snapshot of the current generation for the animation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert optimization vector to readable DataFrame
    df = flat_to_formation(vector, player_names)

    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, 
                  pitch_color='#22312b', line_color='white')

    # Fixed figsize and DPI are critical to ensure all GIF frames have the same dimensions
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor('#22312b')

    # Plot Home Team
    pitch.scatter(df['x'], df['y'], ax=ax, c='blue', s=150, zorder=3, edgecolors='white')

    # Plot Opponents
    if obstacles is not None:
        pitch.scatter(obstacles[:, 0], obstacles[:, 1], ax=ax, c='red', s=80, zorder=3)

    # Plot Ball
    pitch.scatter(ball_pos[0], ball_pos[1], ax=ax, c='yellow', s=200, zorder=5)

    ax.set_title(f"Generation {gen}", color='white', fontsize=16)

    # Save image (bbox_inches='tight' is intentionally omitted to keep size constant)
    filepath = os.path.join(output_dir, f"gen_{gen:04d}.png")
    plt.savefig(filepath, dpi=100)
    plt.close(fig)
    
    return filepath

def create_evolution_gif(frame_dir="code/animation_frames", output="animation.gif"):
    """
    Compiles saved PNG frames into a GIF and cleans up the temporary directory.
    """
    if not os.path.exists(frame_dir):
        print("Animation frames directory not found.")
        return

    # Sort files to ensure correct chronological order
    frames = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")]
    )

    if not frames:
        print("No frames found for animation.")
        return

    try:
        # Read images
        images = [imageio.imread(f) for f in frames]
        
        # Create GIF
        imageio.mimsave(output, images, duration=0.15, loop=0)
        print(f"GIF successfully created: {output}")

        # Cleanup temporary frames
        shutil.rmtree(frame_dir)
        print(f"Cleanup complete: Removed '{frame_dir}'.")

    except Exception as e:
        print(f"Error creating GIF or cleaning up: {e}")