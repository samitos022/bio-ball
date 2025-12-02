# 📘 BIO-BALL (CMA-ES Football Position Optimizer)  
This project was developed as part of a **university assignment** with the goal of applying **evolutionary algorithms** to football analytics.  
The objective is to use **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** to compute **optimal team formations** across the three main phases of play:

- **Offensive possession**  
- **Defensive possession**  
- **Pure defensive phase**

Instead of relying on static positional averages, the project explores how evolutionary optimization can restructure a team’s spatial organization by taking into account tactical constraints, passing quality, ball access, and coverage.  
The result is a tool that allows experimentation on how player arrangements can be improved algorithmically starting from real tracking data.

The project includes:

- Analysis of **Metrica Sports tracking data**  
- Modeling of **tactical constraints** and **spatial penalties**  
- A customizable **multi-term objective function**  
- **CMA-ES** optimization (static and dynamic variants)  
- Visualization of **optimized formations** and **convergence**  

---

## ⚡ How It Works

The optimization pipeline is structured into three main components:

### 1️⃣ Tracking Data Analysis

The project processes tracking and event data from the Metrica Sports sample dataset (`load_data.py`).  
For each frame, the system:

- Identifies **ball possession**  
- Assigns the frame to a **game phase** (offensive, defensive, or non-possession)  
- Computes **average player positions** for each phase (`analysis.py`, `analysis_dynamic.py`)

These averages serve as initial “reference formations” for the optimizer.

---

### 2️⃣ Objective Function

The core of the project is the **fitness function**, modeled as a **weighted sum of multiple tactical components**.  

Each term reflects an element of football decision-making or positional structure.

All weights are defined in `config.py` and can be modified to emphasize different tactical principles.

The individual cost components are:

#### ✔ Structural Constraints (`constraints.py`)  
- Field boundary checks  
- Minimum inter-player distance  
- Preservation of relative ordering (e.g., defenders behind midfielders)  
- Smoothness of the transition from reference to optimized formation  

#### ✔ Field Coverage  
Measured through the Convex Hull: encourages spatially balanced yet expansive structures.

#### ✔ Passing Lanes  
Penalizes:
- Very long passes  
- Unfavorable passing angles  
- Opponent interference on pass lines  

#### ✔ Ball Support  
Encourages at least one teammate to remain close to the ball.

#### ✔ Offside Logic  
Based on ball position and the second-last opponent.

All components are combined through the **weighted-sum fitness model**, making the optimization interpretable and tunable.

---

### 3️⃣ Optimization via CMA-ES

CMA-ES is used to search for the optimal configuration of all 11 players on the field.

Two versions are implemented:

#### 🔵 Static CMA-ES (`main.py`)
- Opponent players keep their **average historical positions**.  
- CMA-ES optimizes the home formation only.  

#### 🔴 Dynamic CMA-ES (`main_dynamic.py`)
- The opponent formation becomes **reactive**, using the logic in `away_reaction.py`:  
  - Lateral and vertical shifting  
  - Role-based adjustments (DEF/MID/FW)  
  - Individual marking  
  - Direct pressing toward the ball  

This creates more realistic optimization scenarios that resemble real match interactions.

---

## 📊 Output Overview

The project produces the following outputs:

- **PNG visualizations** of the optimized formations  
- **Vertical pitch diagrams** for clearer tactical interpretation  
- **CMA-ES convergence plots** showing the evolution of the cost function  
- **GIF animations** that illustrate the optimization process generation by generation  

These outputs help evaluate both the final optimized formation and the evolutionary trajectory that led to it.

---

## ▶️ Running the Project

This section explains how to configure, run, and experiment with the system.

### 1. Install Requirements

Requires **Python ≥ 3.9**.

Install dependencies:

```bash
pip install numpy pandas matplotlib mplsoccer scipy cma imageio
```

### 2. Run Static/Dynamic Optimization

Depending on the type of result you want to obtain, run 

```bash
python main.py
```

for the **static version** or

```bash
python main_dynamic.py
```

for the **dynamic version**.

## 🚀 Author

Developed by Sam Nejati, Daniele Notarangelo e Jasnoor Singh.
