import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
import threading
import io
import sys

from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from utils.analysis_dynamic import plot_formation_with_ball_and_obstacles_2
from utils.away_reaction import react_away_to_home
from utils.reporting import print_fitness_breakdown
from optimization.cma_es import run_optimization as run_cma_static
from optimization.cma_es_dynamic import run_optimization as run_cma_dynamic
from optimization.differential_evolution import run_de_optimization

JSON_FILE = "data/formations/ground_truth.json"
TEMP_IMG = "temp_result.png"


class BioBallGUI:
    def __init__(self, root):
        self.root = root
        root.title("BIO-BALL Optimizer")
        root.geometry("1200x700")
        
        # Variabili
        with open(JSON_FILE) as f:
            self.scenarios = json.load(f)
        self.scenario_var = tk.StringVar(value=list(self.scenarios.keys())[0])
        self.phase_var = tk.StringVar(value="auto")
        self.algo_var = tk.StringVar(value="cma_static")
        self.running = False
        
        # Layout: 2 colonne
        left = tk.Frame(root, width=350, relief=tk.RAISED, borderwidth=1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left.pack_propagate(False)
        
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === LEFT PANEL ===
        tk.Label(left, text="SCENARIO", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        ttk.Combobox(left, textvariable=self.scenario_var, values=list(self.scenarios.keys()), 
                     state="readonly").pack(fill=tk.X, padx=10)
        self.scenario_var.trace('w', lambda *args: self.draw_preview())
        
        # Preview
        self.canvas = tk.Canvas(left, width=330, height=220, bg="#2d5016", highlightthickness=1)
        self.canvas.pack(pady=10, padx=10)
        
        # Fase
        tk.Label(left, text="FASE", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        for text, val in [("Possesso (Auto)", "auto"), ("Difensiva", "def")]:
            tk.Radiobutton(left, text=text, variable=self.phase_var, value=val).pack(anchor=tk.W, padx=20)
        
        # Algoritmo
        tk.Label(left, text="ALGORITMO", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        for text, val in [("CMA-ES Static", "cma_static"), ("CMA-ES Dynamic", "cma_dynamic"), ("Differential Evolution", "de")]:
            tk.Radiobutton(left, text=text, variable=self.algo_var, value=val).pack(anchor=tk.W, padx=20)
        
        # Bottone
        self.btn = tk.Button(left, text="▶ RUN", command=self.run, bg="#4CAF50", fg="white", 
                            font=("Arial", 12, "bold"), height=2)
        self.btn.pack(fill=tk.X, padx=10, pady=20)
        
        # === RIGHT PANEL ===
        # Immagine
        self.img_label = tk.Label(right, text="Risultato apparirà qui", relief=tk.SUNKEN)
        self.img_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Report
        tk.Label(right, text="REPORT", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        scroll = tk.Scrollbar(right)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report = tk.Text(right, height=10, bg="#1e1e1e", fg="#00ff00", font=("Courier", 9))
        self.report.pack(fill=tk.X, side=tk.LEFT, expand=True)
        scroll.config(command=self.report.yview)
        self.report.config(yscrollcommand=scroll.set, state=tk.DISABLED)
        
        self.draw_preview()
    
    def draw_preview(self):
        """Disegna preview campo"""
        self.canvas.delete("all")
        w, h = 330, 220
        data = self.scenarios[self.scenario_var.get()]
        
        # Campo
        self.canvas.create_rectangle(5, 5, w-5, h-5, outline="white", width=2)
        self.canvas.create_line(w/2, 5, w/2, h-5, fill="white", width=1)
        self.canvas.create_oval(w/2-20, h/2-20, w/2+20, h/2+20, outline="white")
        
        # Giocatori
        for pos in data.get("home", {}).values():
            x, y = pos[0] * (w-10) + 5, pos[1] * (h-10) + 5
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", outline="darkred")
        
        for pos in data.get("away", {}).values():
            x, y = pos[0] * (w-10) + 5, pos[1] * (h-10) + 5
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="blue", outline="darkblue")
        
        # Palla
        if "ball" in data:
            bx, by = data["ball"][0] * (w-10) + 5, data["ball"][1] * (h-10) + 5
            self.canvas.create_oval(bx-5, by-5, bx+5, by+5, fill="yellow", outline="orange")
    
    def run(self):
        """Avvia ottimizzazione"""
        if self.running:
            return
        
        self.running = True
        self.btn.config(state=tk.DISABLED)
        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, "Ottimizzazione avviata...\n")
        self.report.config(state=tk.DISABLED)
        
        threading.Thread(target=self.optimize, daemon=True).start()
    
    def optimize(self):
        """Processo ottimizzazione"""
        try:
            scenario = self.scenario_var.get()
            ball_x = self.scenarios[scenario]["ball"][0]
            
            # Determina fase
            if self.phase_var.get() == "def":
                phase_home, phase_away = "Fase difensiva", "Possesso offensivo"
            else:
                phase_away = "Possesso offensivo"
                phase_home = "Possesso offensivo" if ball_x > 0.4 else "Possesso difensivo"
            
            # Setup
            data = setup_scenario(scenario_name=scenario, phase_home=phase_home, phase_away=phase_away)
            
            # Ottimizzazione
            algo = self.algo_var.get()
            if algo == "cma_static":
                best_vec, _ = run_cma_static(data["initial_guess"], data["obstacles_matrix"], 
                                             data["ball_position"], data["starters_home"], phase_home)
                obstacles = data["obstacles_matrix"]
            elif algo == "cma_dynamic":
                best_df, _ = run_cma_dynamic(data["initial_guess"], data["df_away_start"], 
                                            data["ball_position"], data["starters_home"], phase_home)
                best_vec = best_df[['x', 'y']].values.flatten()
                home_df = flat_to_formation(best_vec, data["starters_home"])
                obstacles = react_away_to_home(home_df, data["df_away_start"], data["ball_position"])[["x", "y"]].to_numpy()
            else:  # de
                best_vec, _, _ = run_de_optimization(data["initial_guess"], data["df_away_start"], 
                                                     data["ball_position"], data["starters_home"], phase_home)
                home_df = flat_to_formation(best_vec, data["starters_home"])
                obstacles = react_away_to_home(home_df, data["df_away_start"], data["ball_position"])[["x", "y"]].to_numpy()
            
            # Plot
            best_df = flat_to_formation(best_vec, data["starters_home"])
            plot_formation_with_ball_and_obstacles_2(best_df, f"{phase_home}", "Home", "blue", 
                                                     data["ball_position"], obstacles, TEMP_IMG, show=False)
            
            # Report
            captured = io.StringIO()
            sys.stdout = captured
            print_fitness_breakdown(best_vec, data["starters_home"], obstacles, 
                                   data["ball_position"], data["df_home_start"], phase_home)
            sys.stdout = sys.__stdout__
            
            self.root.after(0, self.show_results, captured.getvalue())
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            self.root.after(0, self.show_error, str(e))
    
    def show_results(self, report):
        """Mostra risultati"""
        self.running = False
        self.btn.config(state=tk.NORMAL)
        
        # Immagine
        try:
            img = Image.open(TEMP_IMG)
            img.thumbnail((700, 400))
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
        except Exception as e:
            self.img_label.config(text=f"Errore immagine: {e}")
        
        # Report
        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, report)
        self.report.config(state=tk.DISABLED)
    
    def show_error(self, msg):
        """Mostra errore"""
        self.running = False
        self.btn.config(state=tk.NORMAL)
        messagebox.showerror("Errore", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = BioBallGUI(root)
    root.mainloop()