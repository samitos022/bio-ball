import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mplsoccer import VerticalPitch
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

# --- PROJECT IMPORTS ---
import config
from utils.setup import setup_scenario
from utils.conversion import flat_to_formation
from utils.away_reaction import react_away_to_home
from optimization.cma_es import run_optimization as run_cma_static
from optimization.cma_es_dynamic import run_optimization as run_cma_dynamic
from optimization.differential_evolution import run_de_optimization
from optimization.constraints import penalty_total
from optimization.cost_functions import (
    cost_coverage, cost_passing_lanes, cost_offside_avoidance,
    cost_marking, cost_defensive_compactness, cost_defensive_line_height,
    cost_ball_pressure, cost_preventive_marking
)

JSON_FILE = "code/data/formations/ground_truth.json"

# Objective descriptions dictionary
OBJECTIVE_DESCRIPTIONS = {
    "Constraints (Hard)": "Hard constraints: player collisions, offside, field boundaries",
    "Coverage": "Weighted coverage of tactical field zones (central > lateral)",
    "Passing Lanes": "Quality and number of available passing options",
    "Offside": "Avoid offside positions in Attacking phase",
    "Marking": "Individual coverage of opponent movement players (excluding goalkeeper)",
    "Compactness": "Team compactness, average distance from centroid (no GK)",
    "Line Height": "Dynamic and elastic defensive line positioning",
    "Ball Pressure": "Distance of nearest player to ball (pressing/support)",
    "Preventive Marking": "Preventive marking of opponents in dangerous zones"
}


class BioBallAppFinal:
    def __init__(self, root):
        self.root = root
        self.root.title("BIO-BALL Optimizer")
        self.root.geometry("1400x900")
        
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#333")
        style.configure("BigButton.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        
        # Card Styles
        style.configure("Card.TFrame", background="#ffffff", relief="raised")
        style.configure("CardTitle.TLabel", font=("Segoe UI", 9), foreground="#666")
        style.configure("CardValue.TLabel", font=("Segoe UI", 18, "bold"), foreground="#1a1a1a")
        
        # Variables
        self.scenarios = self.load_scenarios()
        self.var_scenario = tk.StringVar(value="Historical")
        self.var_phase = tk.StringVar(value="Defensive possession")
        self.var_algo = tk.StringVar(value="cma_static")
        self.is_running = False
        
        # Report data for export
        self.last_report_data = None
        self.last_total_fitness = None
        
        # --- LAYOUT ---
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # 1. SIDEBAR
        self.sidebar = ttk.Frame(main_pane, width=320, padding=15)
        main_pane.add(self.sidebar)
        
        # 2. MAIN AREA
        self.main_area = ttk.Frame(main_pane, padding=10)
        main_pane.add(self.main_area)
        
        self.build_sidebar()
        self.build_main_area()

    def load_scenarios(self):
        base = {"Historical": None}
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, "r") as f:
                    data = json.load(f)
                    base.update(data)
            except: 
                pass
        return base

    def build_sidebar(self):
        ttk.Label(self.sidebar, text="⚙️ CONFIGURATION", style="Header.TLabel").pack(pady=(0, 15), anchor="w")
        
        # Scenario
        ttk.Label(self.sidebar, text="Tactical Scenario:").pack(anchor="w")
        self.combo_scen = ttk.Combobox(self.sidebar, textvariable=self.var_scenario, 
                                       values=list(self.scenarios.keys()), state="readonly")
        self.combo_scen.pack(fill="x", pady=(5, 15))
        self.combo_scen.bind("<<ComboboxSelected>>", self.update_preview)
        
        # Preview Canvas
        self.preview_cv = tk.Canvas(self.sidebar, height=220, bg="#2d5016", 
                                    highlightthickness=2, highlightbackground="#555")
        self.preview_cv.pack(fill="x", pady=(0, 20))
        self.draw_preview()
        
        # Game Phase
        ttk.Label(self.sidebar, text="Game Phase:").pack(anchor="w")
        phases = [
            ("Attacking possession", "Attacking possession"),
            ("Defensive possession", "Defensive possession"),
            ("Defensive phase", "Defensive phase")
        ]
        for text, val in phases:
            ttk.Radiobutton(self.sidebar, text=text, variable=self.var_phase, value=val).pack(anchor="w")
        
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=20)
        
        # Algorithm
        ttk.Label(self.sidebar, text="Optimization Algorithm:").pack(anchor="w")
        algos = [
            ("CMA-ES Static", "cma_static"), 
            ("CMA-ES Dynamic", "cma_dynamic"), 
            ("Diff. Evolution", "de")
        ]
        for text, val in algos:
            ttk.Radiobutton(self.sidebar, text=text, variable=self.var_algo, value=val).pack(anchor="w")
        
        # Run Button
        self.btn_run = ttk.Button(self.sidebar, text="▶ START ANALYSIS", 
                                  style="BigButton.TButton", command=self.start_optimization)
        self.btn_run.pack(fill="x", pady=30, side="bottom")

    def build_main_area(self):
        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.pack(fill="both", expand=True)
        
        # TAB 1: Visualization & Convergence (SIDE BY SIDE)
        self.tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz, text="📊 Results")
        
        # Left frame for pitch
        self.pitch_frame = ttk.Frame(self.tab_viz)
        self.pitch_frame.pack(fill="both", expand=True, side="left", padx=(0, 5))
        
        # Right frame for convergence
        self.conv_frame = ttk.Frame(self.tab_viz)
        self.conv_frame.pack(fill="both", expand=True, side="right", padx=(5, 0))
        
        # TAB 2: Report (IMPROVED)
        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_report, text="📋 Report")
        self.build_report_tab()

    def build_report_tab(self):
        """Build report tab with improved layout"""
        # Main container with scrollbar
        main_container = ttk.Frame(self.tab_report)
        main_container.pack(fill="both", expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_container, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        self.report_content = ttk.Frame(canvas)
        
        self.report_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.report_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Binding mouse wheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # --- SECTION 1: SUMMARY CARDS ---
        cards_frame = ttk.Frame(self.report_content)
        cards_frame.pack(fill="x", padx=20, pady=20)
        
        self.card_fitness = self.create_summary_card(cards_frame, "Total Fitness", "---", 0)
        self.card_phase = self.create_summary_card(cards_frame, "Game Phase", "---", 1)
        self.card_algo = self.create_summary_card(cards_frame, "Algorithm", "---", 2)
        
        # --- SECTION 1.5: ALGORITHM PARAMETERS ---
        params_frame = ttk.LabelFrame(self.report_content, text="Algorithm Parameters", 
                                      padding=15)
        params_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Create text widget for parameters
        self.params_text = tk.Text(params_frame, height=6, font=("Consolas", 9), 
                                   bg="#f9f9f9", relief="flat", wrap="word")
        self.params_text.pack(fill="x", expand=True)
        self.params_text.config(state="disabled")  # Read-only
        
        # --- SECTION 2: CHARTS ---
        charts_frame = ttk.Frame(self.report_content)
        charts_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Frame for pie chart
        self.pie_frame = ttk.Frame(charts_frame, relief="solid", borderwidth=1)
        self.pie_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Frame for bar chart
        self.bar_frame = ttk.Frame(charts_frame, relief="solid", borderwidth=1)
        self.bar_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        # --- SECTION 3: TREEVIEW ---
        tree_container = ttk.Frame(self.report_content)
        tree_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Header with export button
        header = ttk.Frame(tree_container)
        header.pack(fill="x", pady=(0, 10))
        
        ttk.Label(header, text="Objective Details", 
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        
        self.btn_export = ttk.Button(header, text="📥 Export Report", 
                                     command=self.export_report)
        self.btn_export.pack(side="right")
        self.btn_export.config(state="disabled")
        
        # Improved Treeview (WITHOUT Description column)
        cols = ("Value", "Weight", "Cost", "Impact")
        self.tree = ttk.Treeview(tree_container, columns=cols, show="tree headings", height=15)
        
        self.tree.heading("#0", text="Objective", anchor="w")
        self.tree.column("#0", width=250)
        
        col_widths = [120, 100, 120, 100]
        for i, (col, width) in enumerate(zip(cols, col_widths)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center")
        
        # Scrollbar
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        
        # Configure tags for colors
        self.tree.tag_configure('critical', background='#fee', foreground='#c00')
        self.tree.tag_configure('warning', background='#ffc', foreground='#c60')
        self.tree.tag_configure('good', background='#efe', foreground='#060')
        self.tree.tag_configure('total', background='#ddd', font=("Segoe UI", 10, "bold"))
        self.tree.tag_configure('detail', foreground='#666')

    def create_summary_card(self, parent, title, value, column):
        """Create a summary card"""
        card = ttk.Frame(parent, relief="raised", borderwidth=2)
        card.grid(row=0, column=column, sticky="ew", padx=5)
        parent.columnconfigure(column, weight=1)
        
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(pady=(10, 5))
        label_value = ttk.Label(card, text=value, style="CardValue.TLabel")
        label_value.pack(pady=(0, 10))
        
        return label_value

    def draw_preview(self):
        self.preview_cv.delete("all")
        self.preview_cv.update_idletasks() 
        w = self.preview_cv.winfo_width()
        if w < 10: 
            w = 280
        h = 220
        
        pad = 5
        # Field lines
        self.preview_cv.create_rectangle(pad, pad, w-pad, h-pad, outline="white", width=2)
        self.preview_cv.create_line(w/2, pad, w/2, h-pad, fill="white", width=1)
        self.preview_cv.create_oval(w/2-25, h/2-25, w/2+25, h/2+25, outline="white", width=1)
        
        # Penalty areas
        penalty_h = h * 0.55
        penalty_w = w * 0.16
        penalty_top = (h - penalty_h) / 2
        penalty_bot = penalty_top + penalty_h
        
        self.preview_cv.create_rectangle(pad, penalty_top, pad+penalty_w, penalty_bot, 
                                        outline="white", width=1)
        self.preview_cv.create_rectangle(w-pad-penalty_w, penalty_top, w-pad, penalty_bot, 
                                        outline="white", width=1)
        
        # Goal areas
        goal_h = h * 0.25
        goal_w = w * 0.05
        goal_top = (h - goal_h) / 2
        goal_bot = goal_top + goal_h
        
        self.preview_cv.create_rectangle(pad, goal_top, pad+goal_w, goal_bot, 
                                        outline="white", width=1)
        self.preview_cv.create_rectangle(w-pad-goal_w, goal_top, w-pad, goal_bot, 
                                        outline="white", width=1)
        
        # Player data
        name = self.var_scenario.get()
        if name == "Historical" or name not in self.scenarios:
            self.preview_cv.create_text(w/2, h/2, text="Historical Data\n(Match Average)", 
                                       fill="white", font=("Arial", 10, "bold"), justify="center")
            return
        
        data = self.scenarios[name]
        
        def draw_token(pos, color, outline="white", r=4):
            cx, cy = pos[0]*w, pos[1]*h
            cx = max(pad, min(w-pad, cx))
            cy = max(pad, min(h-pad, cy))
            self.preview_cv.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline=outline)
        
        for pos in data.get("home", {}).values():
            draw_token(pos, "#ff4444")
        for pos in data.get("away", {}).values():
            draw_token(pos, "#4444ff")
        if "ball" in data:
            draw_token(data["ball"], "yellow", "black", r=5)

    def update_preview(self, event):
        self.draw_preview()

    def start_optimization(self):
        if self.is_running: 
            return
        self.is_running = True
        self.btn_run.config(state="disabled", text="⏳ PROCESSING...")
        
        # Clear everything
        for widget in self.pitch_frame.winfo_children(): 
            widget.destroy()
        for widget in self.conv_frame.winfo_children(): 
            widget.destroy()
        self.tree.delete(*self.tree.get_children())
        
        # Clear report charts
        for widget in self.pie_frame.winfo_children():
            widget.destroy()
        for widget in self.bar_frame.winfo_children():
            widget.destroy()
        
        scenario = self.var_scenario.get()
        phase = self.var_phase.get()
        algo = self.var_algo.get()
        threading.Thread(target=self.run_thread, args=(scenario, phase, algo), daemon=True).start()

    def run_thread(self, scenario, phase, algo):
        try:
            scenario_key = None if scenario == "Historical" else scenario
            
            phase_away = "Defensive phase" if "possession" in phase else "Attacking possession"
            
            data = setup_scenario(scenario_name=scenario_key, phase_home=phase, phase_away=phase_away)

            best_vec, cost_history, final_obstacles = None, [], None
            
            if algo == "cma_static":
                best_vec, cost_history = run_cma_static(
                    data["initial_guess"], data["obstacles_matrix"], 
                    data["ball_position"], data["starters_home"], phase
                )
                final_obstacles = data["obstacles_matrix"]
            elif algo == "cma_dynamic":
                best_vec, cost_history = run_cma_dynamic(
                    data["initial_guess"], data["df_away_start"], 
                    data["ball_position"], data["starters_home"], phase
                )
                home_df = flat_to_formation(best_vec, data["starters_home"])
                away_df = react_away_to_home(home_df, data["df_away_start"], data["ball_position"])
                final_obstacles = away_df[["x", "y"]].to_numpy()
            elif algo == "de":
                best_vec, _, cost_history = run_de_optimization(
                    data["initial_guess"], data["df_away_start"],
                    data["ball_position"], data["starters_home"], phase
                )
                final_obstacles = data["obstacles_matrix"]
            
            report_data = self.calculate_report(best_vec, data, final_obstacles, phase)
            self.root.after(0, self.on_finish, best_vec, cost_history, data, final_obstacles, 
                          report_data, phase, algo)
            
        except Exception as e:
            self.root.after(0, self.on_error, str(e))

    def calculate_report(self, vector, data, obstacles, phase):
        df = flat_to_formation(vector, data["starters_home"])
        ball = data["ball_position"]
        ref = data["df_home_start"]
        weights = config.PHASE_WEIGHTS.get(phase, config.PHASE_WEIGHTS["Defensive phase"])
        report = []
        
        res_c = penalty_total({"Start": ref, "Candidate": df}, detailed=True)
        report.append({
            "name": "Constraints (Hard)", 
            "raw": res_c["total"], 
            "weight": config.OBJ_W_CONSTRAINTS, 
            "cost": res_c["total"] * config.OBJ_W_CONSTRAINTS,
            "details": {k: v for k, v in res_c.items() if k != "total" and v > 0}
        })
        
        tasks = [
            ("W_COVERAGE", cost_coverage, "Coverage", [df], {"detailed": True}),
            ("W_PASSING", cost_passing_lanes, "Passing Lanes", [df, obstacles, ball], 
             {"phase_type": phase, "detailed": True}),
            ("W_OFFSIDE", cost_offside_avoidance, "Offside", [df, obstacles, ball], 
             {"detailed": True}),
            ("W_MARKING", cost_marking, "Marking", [df, obstacles], {"detailed": True}),
            ("W_COMPACTNESS", cost_defensive_compactness, "Compactness", [df], 
             {"detailed": True}),
            ("W_LINE_HEIGHT", cost_defensive_line_height, "Line Height", [df, ball], 
             {"detailed": True}),
            ("W_BALL_PRESS", cost_ball_pressure, "Ball Pressure", [df, ball], 
             {"detailed": True}),
            ("W_PREV_MARKING", cost_preventive_marking, "Preventive Marking", 
             [df, obstacles], {"detailed": True}),
        ]
        
        for w_key, func, label, args, kwargs in tasks:
            w = weights.get(w_key, 0)
            if w > 0:
                res = func(*args, **kwargs)
                dets = {k: v for k, v in res.items() if k != "total"}
                report.append({
                    "name": label, 
                    "raw": res["total"], 
                    "weight": w, 
                    "cost": res["total"] * w, 
                    "details": dets
                })
        
        return report

    def get_impact_emoji(self, impact):
        """Return emoji based on impact"""
        if impact > 20:
            return "🔴"  # Critical
        elif impact > 10:
            return "🟡"  # Warning
        else:
            return "🟢"  # Good

    def get_impact_tag(self, impact):
        """Return color tag based on impact"""
        if impact > 20:
            return "critical"
        elif impact > 10:
            return "warning"
        else:
            return "good"

    def get_algorithm_params(self, algo):
        """Get algorithm parameters from config"""
        if algo in ["cma_static", "cma_dynamic"]:
            return {
                "Max Iterations": config.CMA_MAXITER,
                "Population Size": config.CMA_POPSIZE,
                "Initial Sigma": config.CMA_SIGMA_INIT,
                "Tolerance (Fun)": config.CMA_TOLFUN
            }
        elif algo == "de":
            return {
                "Max Iterations": config.DE_MAXITER,
                "Population Size": config.DE_POPSIZE,
                "Mutation Range": f"{config.DE_MUTATION[0]} - {config.DE_MUTATION[1]}",
                "Recombination": config.DE_RECOMBINATION,
                "Tolerance": config.DE_TOL
            }
        return {}

    def update_params_display(self, algo):
        """Update algorithm parameters display"""
        params = self.get_algorithm_params(algo)
        
        self.params_text.config(state="normal")
        self.params_text.delete(1.0, tk.END)
        
        # Format parameters nicely
        text = ""
        for key, value in params.items():
            text += f"{key:.<25} {value}\n"
        
        self.params_text.insert(1.0, text)
        self.params_text.config(state="disabled")

    def on_finish(self, best_vec, cost_history, data, obstacles, report_data, phase, algo):
        self.btn_run.config(state="normal", text="▶ START ANALYSIS")
        self.is_running = False
        
        # Save data for export
        self.last_report_data = report_data
        self.last_phase = phase
        self.last_algo = algo
        
        # Force update frames to get correct dimensions
        self.pitch_frame.update()
        self.conv_frame.update()
        
        best_df = flat_to_formation(best_vec, data["starters_home"])
        
        # 1. PITCH (with proper sizing for side-by-side layout)
        plt.close('all')
        
        # Get actual frame dimensions
        frame_width = self.pitch_frame.winfo_width()
        frame_height = self.pitch_frame.winfo_height()
        
        # Calculate figure size for vertical pitch (taller than wide)
        fig_width = max(5, frame_width / 100 - 1)
        fig_height = max(7, frame_height / 100 - 1)
        
        fig_pitch = plt.Figure(figsize=(fig_width, fig_height), dpi=100)
        ax_p = fig_pitch.add_subplot(111)
        pitch = VerticalPitch(pitch_type='metricasports', pitch_length=105, pitch_width=68, 
                              pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax_p)
        
        pitch.scatter(best_df.x, best_df.y, ax=ax_p, c='#1E90FF', s=200, 
                     edgecolors='white', lw=2, zorder=3, label='Home')
        if obstacles is not None:
            pitch.scatter(obstacles[:, 0], obstacles[:, 1], ax=ax_p, c='#555555', 
                         s=150, alpha=0.7, edgecolors='#888', zorder=2)
        ball = data["ball_position"]
        pitch.scatter(ball[0], ball[1], ax=ax_p, c='#FFD700', s=160, 
                     edgecolors='black', lw=2, zorder=4)
        
        ax_p.set_title(f"Result: {phase}", fontsize=12, fontweight='bold')
        fig_pitch.tight_layout()
        
        canvas_p = FigureCanvasTkAgg(fig_pitch, master=self.pitch_frame)
        canvas_p.draw()
        canvas_p.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. CONVERGENCE (with proper sizing for side-by-side layout)
        frame_width = self.conv_frame.winfo_width()
        frame_height = self.conv_frame.winfo_height()
        
        # Calculate figure size for convergence chart (wider than tall)
        fig_width = max(6, frame_width / 100 - 1)
        fig_height = max(7, frame_height / 100 - 1)
        
        fig_conv = plt.Figure(figsize=(fig_width, fig_height), dpi=100)
        ax_c = fig_conv.add_subplot(111)
        ax_c.plot(cost_history, color='#2E8B57', linewidth=2)
        ax_c.set_title("Fitness Convergence", fontsize=12, fontweight='bold')
        ax_c.set_xlabel("Generation")
        ax_c.set_ylabel("Cost")
        ax_c.grid(True, alpha=0.3)
        fig_conv.tight_layout()
        
        canvas_c = FigureCanvasTkAgg(fig_conv, master=self.conv_frame)
        canvas_c.draw()
        canvas_c.get_tk_widget().pack(fill="both", expand=True)
        
        # 3. IMPROVED REPORT
        total_fitness = sum(item["cost"] for item in report_data)
        self.last_total_fitness = total_fitness
        
        # Update Summary Cards
        self.card_fitness.config(text=f"{total_fitness:.3f}")
        self.card_phase.config(text=phase)
        
        algo_names = {
            "cma_static": "CMA-ES Static",
            "cma_dynamic": "CMA-ES Dynamic",
            "de": "Diff. Evolution"
        }
        self.card_algo.config(text=algo_names.get(algo, algo))
        
        # Update algorithm parameters display
        self.update_params_display(algo)
        
        # 3.1 PIE CHART
        self.draw_pie_chart(report_data, total_fitness)
        
        # 3.2 BAR CHART
        self.draw_bar_chart(report_data)
        
        # 3.3 TREEVIEW (WITHOUT Description column)
        for item in report_data:
            impact = (item["cost"] / total_fitness * 100) if total_fitness > 0 else 0
            emoji = self.get_impact_emoji(impact)
            tag = self.get_impact_tag(impact)
            
            row_id = self.tree.insert("", "end", 
                text=f"{emoji} {item['name']}", 
                values=(
                    f"{item['raw']:.2f}",
                    f"{item['weight']:.1f}",
                    f"{item['cost']:.3f}",
                    f"{impact:.1f}%"
                ),
                tags=(tag,)
            )
            
            # Details
            if "details" in item and item["details"]:
                for k, v in item["details"].items():
                    if isinstance(v, (int, float)):
                        val = f"{v:.3f}"
                    elif isinstance(v, list):
                        val = ", ".join(str(x) for x in v[:3])
                        if len(v) > 3:
                            val += f" ... (+{len(v)-3})"
                    else:
                        val = str(v)
                    
                    self.tree.insert(row_id, "end", 
                        text="", 
                        values=(f"  ↳ {k}", val, "-", "-"),
                        tags=('detail',)
                    )
        
        # Total row
        self.tree.insert("", "end", 
            text="", 
            values=(
                "TOTAL FITNESS",
                "-",
                f"{total_fitness:.3f}",
                "100%"
            ),
            tags=('total',)
        )
        
        # Enable export
        self.btn_export.config(state="normal")
        
        # Focus on results tab
        self.notebook.select(self.tab_viz)

    def draw_pie_chart(self, report_data, total_fitness):
        """Draw pie chart of impact"""
        for widget in self.pie_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(111)
        
        # Prepare data
        labels = []
        sizes = []
        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6']
        
        for item in report_data:
            impact = (item["cost"] / total_fitness * 100) if total_fitness > 0 else 0
            if impact > 0.5:  # Show only significant objectives
                labels.append(item["name"])
                sizes.append(impact)
        
        # Draw
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors[:len(sizes)],
                                           startangle=90, textprops={'fontsize': 9})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title("Objective Impact Distribution", fontsize=11, fontweight='bold', pad=20)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.pie_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def draw_bar_chart(self, report_data):
        """Draw bar chart of costs"""
        for widget in self.bar_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(111)
        
        # Prepare data
        names = [item["name"] for item in report_data]
        costs = [item["cost"] for item in report_data]
        
        # Colors based on cost
        max_cost = max(costs) if costs else 1
        colors = ['#ef4444' if c > max_cost * 0.5 else 
                 '#f59e0b' if c > max_cost * 0.2 else 
                 '#10b981' for c in costs]
        
        # Draw horizontal bars
        y_pos = np.arange(len(names))
        ax.barh(y_pos, costs, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Cost", fontsize=10)
        ax.set_title("Costs per Objective", fontsize=11, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.bar_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def export_report(self):
        """Export report to CSV format"""
        if not self.last_report_data:
            messagebox.showwarning("Warning", "No report available for export.")
            return
        
        # Dialog to save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"bioball_report_{timestamp}.csv"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(["BIO-BALL Optimization Report"])
                writer.writerow([f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                writer.writerow([f"Phase: {self.last_phase}"])
                writer.writerow([f"Algorithm: {self.last_algo}"])
                writer.writerow([f"Total Fitness: {self.last_total_fitness:.4f}"])
                writer.writerow([])
                
                # Algorithm parameters
                writer.writerow(["Algorithm Parameters"])
                params = self.get_algorithm_params(self.last_algo)
                for key, value in params.items():
                    writer.writerow([f"  {key}", value])
                writer.writerow([])
                
                # Objectives table
                writer.writerow(["Objective", "Description", "Raw Value", "Weight", 
                               "Total Cost", "Impact %"])
                
                for item in self.last_report_data:
                    impact = (item["cost"] / self.last_total_fitness * 100) if self.last_total_fitness > 0 else 0
                    description = OBJECTIVE_DESCRIPTIONS.get(item["name"], "")
                    
                    writer.writerow([
                        item["name"],
                        description,
                        f"{item['raw']:.4f}",
                        f"{item['weight']:.2f}",
                        f"{item['cost']:.4f}",
                        f"{impact:.2f}%"
                    ])
                    
                    # Details
                    if "details" in item and item["details"]:
                        for k, v in item["details"].items():
                            val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                            writer.writerow([f"  - {k}", "", val, "", "", ""])
                
                writer.writerow([])
                writer.writerow(["TOTAL", "", "", "", f"{self.last_total_fitness:.4f}", "100%"])
            
            messagebox.showinfo("Success", f"Report exported successfully:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during export:\n{str(e)}")

    def on_error(self, msg):
        self.btn_run.config(state="normal", text="▶ START ANALYSIS")
        self.is_running = False
        messagebox.showerror("Error", msg)


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: 
        pass
    
    root = tk.Tk()
    app = BioBallAppFinal(root)
    root.mainloop()