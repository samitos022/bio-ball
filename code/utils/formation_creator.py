import tkinter as tk
from tkinter import simpledialog, messagebox
import json
import os

# Configurazione Campo (Pixel per la GUI)
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 520 # Proporzione circa 105x68
PLAYER_RADIUS = 10
OUTPUT_FILE = "code/data/formations/ground_truth.json"

class FormationCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Bio-Ball Formation Creator")
        
        # Struttura dati per i giocatori
        # Usiamo indici 0-10 che corrisponderanno all'ordine nella lista 'starters_home'
        self.players = {} 
        self.drag_data = {"x": 0, "y": 0, "item": None}

        # Creazione Cartella Dati
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        # --- GUI LAYOUT ---
        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Nome Formazione:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(top_frame, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(top_frame, text="SALVA su JSON", command=self.save_formation, bg="#4CAF50", fg="white")
        save_btn.pack(side=tk.LEFT, padx=10)
        
        help_lbl = tk.Label(top_frame, text="(Trascina i pallini e salva)", fg="gray")
        help_lbl.pack(side=tk.LEFT)

        # Canvas Campo
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#2E8B57")
        self.canvas.pack(padx=10, pady=10)

        self.draw_pitch()
        self.init_players()

        # Binding Eventi Mouse
        self.canvas.tag_bind("player", "<ButtonPress-1>", self.on_token_press)
        self.canvas.tag_bind("player", "<ButtonRelease-1>", self.on_token_release)
        self.canvas.tag_bind("player", "<B1-Motion>", self.on_token_motion)

    def draw_pitch(self):
        """Disegna linee bianche del campo"""
        # Bordo
        self.canvas.create_rectangle(10, 10, CANVAS_WIDTH-10, CANVAS_HEIGHT-10, outline="white", width=2)
        # Metà campo
        self.canvas.create_line(CANVAS_WIDTH/2, 10, CANVAS_WIDTH/2, CANVAS_HEIGHT-10, fill="white", width=2)
        self.canvas.create_oval(CANVAS_WIDTH/2-50, CANVAS_HEIGHT/2-50, CANVAS_WIDTH/2+50, CANVAS_HEIGHT/2+50, outline="white", width=2)
        
        # Aree rigore (approssimative)
        # Sinistra
        self.canvas.create_rectangle(10, CANVAS_HEIGHT/2-100, 130, CANVAS_HEIGHT/2+100, outline="white", width=2)
        # Destra
        self.canvas.create_rectangle(CANVAS_WIDTH-130, CANVAS_HEIGHT/2-100, CANVAS_WIDTH-10, CANVAS_HEIGHT/2+100, outline="white", width=2)

    def init_players(self):
        """Crea gli 11 pallini in posizione di default"""
        # Posizione iniziale 4-4-2 standard (normalizzata e convertita in pixel)
        # Coordinate normalizzate (0-1)
        default_positions = [
            (0.05, 0.5), # GK
            (0.20, 0.2), (0.20, 0.4), (0.20, 0.6), (0.20, 0.8), # Difesa
            (0.40, 0.2), (0.40, 0.4), (0.40, 0.6), (0.40, 0.8), # Centrocampo
            (0.60, 0.4), (0.60, 0.6)  # Attacco
        ]

        for i, (nx, ny) in enumerate(default_positions):
            cx = nx * CANVAS_WIDTH
            cy = ny * CANVAS_HEIGHT
            
            # Colore diverso per il portiere (indice 0)
            color = "yellow" if i == 0 else "red"
            
            # Crea cerchio
            item_id = self.canvas.create_oval(
                cx - PLAYER_RADIUS, cy - PLAYER_RADIUS,
                cx + PLAYER_RADIUS, cy + PLAYER_RADIUS,
                fill=color, outline="black", tags=("player",)
            )
            
            # Crea numero
            text_id = self.canvas.create_text(cx, cy, text=str(i+1), tags=("player",))
            
            self.players[item_id] = {"id": i, "text_id": text_id}
            self.players[text_id] = {"id": i, "oval_id": item_id} # Map reverse per comodità

    # --- DRAG & DROP LOGIC ---
    def on_token_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        # Assicuriamoci di prendere l'oval, non il testo
        if item not in self.players: return 
        
        # Se abbiamo cliccato il testo, prendiamo l'ovale collegato
        if "oval_id" in self.players[item]:
            item = self.players[item]["oval_id"]

        self.drag_data["item"] = item
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_token_release(self, event):
        self.drag_data["item"] = None
        self.drag_data["x"] = 0
        self.drag_data["y"] = 0

    def on_token_motion(self, event):
        if self.drag_data["item"]:
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            oval = self.drag_data["item"]
            text = self.players[oval]["text_id"]
            
            # Muovi cerchio e testo
            self.canvas.move(oval, dx, dy)
            self.canvas.move(text, dx, dy)
            
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    # --- SAVING LOGIC ---
    def save_formation(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Attenzione", "Inserisci un nome per la formazione!")
            return

        # Raccogli coordinate normalizzate (0-1)
        formation_coords = {}
        
        # Itera sugli items nel canvas
        # Nota: self.players ha entrate doppie (id->dict e text_id->dict).
        # Filtriamo per prendere solo gli ovali.
        
        items_processed = []
        
        for item_id, data in self.players.items():
            if "text_id" in data: # È un ovale
                idx = data["id"]
                coords = self.canvas.coords(item_id)
                # coords è [x1, y1, x2, y2]
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                
                # Normalizza
                norm_x = cx / CANVAS_WIDTH
                norm_y = cy / CANVAS_HEIGHT
                
                # Salva con indice stringa "0", "1"...
                formation_coords[str(idx)] = [round(norm_x, 4), round(norm_y, 4)]

        # Carica JSON esistente se c'è
        full_data = {}
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, "r") as f:
                    full_data = json.load(f)
            except:
                pass # File corrotto o vuoto

        # Aggiungi/Sovrascrivi formazione
        full_data[name] = formation_coords

        # Salva
        with open(OUTPUT_FILE, "w") as f:
            json.dump(full_data, f, indent=4)

        messagebox.showinfo("Successo", f"Formazione '{name}' salvata in {OUTPUT_FILE}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FormationCreator(root)
    root.mainloop()