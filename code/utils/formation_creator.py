import tkinter as tk
from tkinter import simpledialog, messagebox
import json
import os

# Configurazione Campo
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 520 
PLAYER_RADIUS = 10
BALL_RADIUS = 6
OUTPUT_FILE = "code/data/formations/ground_truth.json"

class FormationCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Bio-Ball Scenario Creator (Home, Away, Ball)")
        
        # Struttura dati interna
        # items_map: id_canvas -> {type: "home"|"away"|"ball", id: index, text_id: id}
        self.items_map = {} 
        self.drag_data = {"x": 0, "y": 0, "item": None}

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        # --- GUI LAYOUT ---
        top_frame = tk.Frame(root, bg="#ddd")
        top_frame.pack(fill=tk.X, padx=0, pady=0)
        
        inner_frame = tk.Frame(top_frame, bg="#ddd")
        inner_frame.pack(padx=10, pady=10)

        tk.Label(inner_frame, text="Nome Scenario:", bg="#ddd").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(inner_frame, width=25)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(inner_frame, text="💾 SALVA SCENARIO", command=self.save_scenario, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        save_btn.pack(side=tk.LEFT, padx=15)
        
        # Legenda
        legend_frame = tk.Frame(root)
        legend_frame.pack(pady=5)
        tk.Label(legend_frame, text="🔴 Casa (Target)", fg="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🔵 Avversari", fg="blue", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🟡 Palla", fg="#DAA520", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#2E8B57")
        self.canvas.pack(padx=10, pady=10)

        self.draw_pitch()
        self.init_entities()

        # Binding Mouse
        self.canvas.tag_bind("movable", "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind("movable", "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind("movable", "<B1-Motion>", self.on_motion)

    def draw_pitch(self):
        # Disegno campo classico
        self.canvas.create_rectangle(10, 10, CANVAS_WIDTH-10, CANVAS_HEIGHT-10, outline="white", width=2)
        self.canvas.create_line(CANVAS_WIDTH/2, 10, CANVAS_WIDTH/2, CANVAS_HEIGHT-10, fill="white", width=2)
        self.canvas.create_oval(CANVAS_WIDTH/2-50, CANVAS_HEIGHT/2-50, CANVAS_WIDTH/2+50, CANVAS_HEIGHT/2+50, outline="white", width=2)
        self.canvas.create_rectangle(10, CANVAS_HEIGHT/2-100, 130, CANVAS_HEIGHT/2+100, outline="white", width=2)
        self.canvas.create_rectangle(CANVAS_WIDTH-130, CANVAS_HEIGHT/2-100, CANVAS_WIDTH-10, CANVAS_HEIGHT/2+100, outline="white", width=2)

    def create_token(self, nx, ny, color, label, type_name, idx):
        cx = nx * CANVAS_WIDTH
        cy = ny * CANVAS_HEIGHT
        radius = BALL_RADIUS if type_name == "ball" else PLAYER_RADIUS
        
        item_id = self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=color, outline="black", width=1, tags=("movable",)
        )
        
        text_id = None
        if label:
            text_id = self.canvas.create_text(cx, cy, text=label, fill="white", tags=("movable",))
        
        self.items_map[item_id] = {
            "type": type_name, 
            "id": idx, 
            "text_id": text_id,
            "oval_id": item_id # self reference
        }
        if text_id:
            self.items_map[text_id] = {
                "type": type_name, 
                "id": idx, 
                "oval_id": item_id
            }

    def init_entities(self):
        # 1. PALLA (Centro campo)
        self.create_token(0.5, 0.5, "yellow", "⚽", "ball", 0)

        # 2. SQUADRA DI CASA (Red - 4-4-2 default) - Quelli che ottimizzi
        home_pos = [
            (0.05, 0.5), # GK
            (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8), # Def
            (0.45, 0.2), (0.45, 0.4), (0.45, 0.6), (0.45, 0.8), # Mid
            (0.65, 0.4), (0.65, 0.6)  # Att
        ]
        for i, (nx, ny) in enumerate(home_pos):
            self.create_token(nx, ny, "red", str(i+1), "home", i)

        # 3. AVVERSARI (Blue - 4-4-2 speculare default) - Gli ostacoli
        away_pos = [
            (0.95, 0.5), # GK Opp
            (0.75, 0.2), (0.75, 0.4), (0.75, 0.6), (0.75, 0.8),
            (0.55, 0.2), (0.55, 0.4), (0.55, 0.6), (0.55, 0.8),
            (0.35, 0.4), (0.35, 0.6)
        ]
        for i, (nx, ny) in enumerate(away_pos):
            self.create_token(nx, ny, "blue", str(i+1), "away", i)

    # --- DRAG & DROP ---
    def on_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        if item not in self.items_map: return
        
        # Se clicco sul testo, prendo l'oval
        if "oval_id" in self.items_map[item]:
            item = self.items_map[item]["oval_id"]

        self.drag_data["item"] = item
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_release(self, event):
        self.drag_data["item"] = None

    def on_motion(self, event):
        if self.drag_data["item"]:
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            oval = self.drag_data["item"]
            data = self.items_map[oval]
            
            self.canvas.move(oval, dx, dy)
            if data["text_id"]:
                self.canvas.move(data["text_id"], dx, dy)
            
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    # --- SAVE ---
    def save_scenario(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Error", "Inserisci un nome per lo scenario!")
            return

        scenario_data = {
            "home": {},
            "away": {},
            "ball": []
        }

        # Estrai coordinate
        for item_id, data in self.items_map.items():
            if "text_id" in data: # È un oval (token principale)
                coords = self.canvas.coords(item_id)
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                nx = round(cx / CANVAS_WIDTH, 4)
                ny = round(cy / CANVAS_HEIGHT, 4)
                
                t = data["type"]
                idx = str(data["id"])

                if t == "ball":
                    scenario_data["ball"] = [nx, ny]
                else:
                    scenario_data[t][idx] = [nx, ny]

        # Salva su file
        full_db = {}
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, "r") as f:
                    full_db = json.load(f)
            except: pass

        full_db[name] = scenario_data

        with open(OUTPUT_FILE, "w") as f:
            json.dump(full_db, f, indent=4)
        
        messagebox.showinfo("Salvato", f"Scenario '{name}' salvato con successo!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FormationCreator(root)
    root.mainloop()