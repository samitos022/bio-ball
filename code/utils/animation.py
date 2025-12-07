import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os
import shutil  # <--- Necessario per cancellare cartelle piene
from utils.conversion import flat_to_formation
import imageio.v2 as imageio  # Usa v2 per evitare warning deprecati

def save_generation_plot(vector, player_names, obstacles, ball_pos, gen, output_dir="code/animation_frames"):
    # Crea cartella
    os.makedirs(output_dir, exist_ok=True)

    # Converti vettore → dataframe
    df = flat_to_formation(vector, player_names)

    pitch = Pitch(pitch_type='metricasports', pitch_length=106, pitch_width=68, pitch_color='#22312b', line_color='white')

    # Fissare dimensioni e DPI è importante per evitare errori durante la creazione della GIF
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

    # Salva immagine (senza bbox_inches='tight' per mantenere dimensioni costanti)
    filepath = os.path.join(output_dir, f"gen_{gen:04d}.png")
    plt.savefig(filepath, dpi=100)
    plt.close(fig)
    
    return filepath

def create_evolution_gif(frame_dir="code/animation_frames", output="animation.gif"):
    # Controllo se la cartella esiste
    if not os.path.exists(frame_dir):
        print("Cartella frame non trovata, impossibile creare GIF.")
        return

    frames = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")]
    )

    if not frames:
        print("Nessun frame trovato.")
        return

    try:
        # Legge le immagini
        images = [imageio.imread(f) for f in frames]
        
        # Crea la GIF
        imageio.mimsave(output, images, duration=0.15, loop=0)
        print(f"GIF creata con successo: {output}")

        # --- PULIZIA ---
        # Cancella la cartella e tutto il suo contenuto
        shutil.rmtree(frame_dir)
        print(f"Pulizia completata: cartella '{frame_dir}' eliminata.")

    except Exception as e:
        print(f"Errore durante la creazione della GIF o la pulizia: {e}")