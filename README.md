# 📘 BIO-BALL (Evolutionary Football Formation Optimization)

BIO-BALL is a **university research project** focused on applying **evolutionary optimization algorithms** to football tactical analysis.
The objective is to automatically compute **optimal team formations** across different **phases of play**, starting from real tracking data.

Instead of relying on static positional averages, the project explores how evolutionary optimization can **reshape a team’s spatial organization** by incorporating tactical constraints, passing quality, ball access, offside logic, and defensive coverage.

The project is built around **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**, with additional support for **Differential Evolution**, and allows experimentation with both **static** and **reactive (dynamic)** opponent behavior.

---

## ⚽ Phases of Play

The optimizer can be applied to different game contexts:

* **Offensive possession** – structured attacking shape and passing options
* **Defensive possession** – compactness, coverage, and marking
* **Non-possession / pure defensive phase** – block positioning and space denial

Each phase produces a distinct reference formation and optimization objective.

---

## 🧩 Project Components

### 📊 Tracking Data Analysis

Tracking and event data from the **Metrica Sports open dataset** are processed using `load_data.py` and `analysis.py`.
For each frame, the system:

* Detects **ball possession**
* Assigns the frame to a **game phase**
* Computes **average player positions** per phase

These averages act as **reference formations** that initialize the evolutionary search.

---

### 🎯 Objective Function

The optimization problem is defined through a **multi-term weighted cost function**, designed to remain interpretable and tunable.

All weights are configurable and can be optimized or manually adjusted.

#### Main Cost Components

* **Structural Constraints**

  * Field boundary enforcement
  * Minimum inter-player distance
  * Role ordering (DEF < MID < FW)
  * Smooth deviation from reference formation

* **Field Coverage**

  * Encouraged via **Convex Hull area** control

* **Passing Lanes**

  * Penalization of long or poorly angled passes
  * Opponent interference on passing lines

* **Ball Support**

  * Ensures proximity of at least one teammate to the ball

* **Offside Logic**

  * Based on ball position and second-last opponent

The final fitness is computed as a **weighted sum**, making tactical trade-offs explicit.

---

## 🧠 Optimization Algorithms

### 🔵 Static CMA-ES

* Implemented in `optimization/cma_es.py`
* Opponent players remain fixed at historical average positions
* Optimizes only the home team formation

### 🔴 Dynamic CMA-ES

* Opponent becomes **reactive** through `away_reaction.py`
* Includes:

  * Lateral and vertical shifting
  * Role-based movements (DEF / MID / FW)
  * Individual marking
  * Ball-oriented pressing

This setup produces more realistic tactical interactions.

### 🟢 Differential Evolution (Benchmark)

* Implemented in `optimization/differential_evolution.py`
* Used **exclusively as a final evaluation benchmark**
* Provides a comparative baseline against CMA-ES solutions
* Not intended for tactical tuning, but for assessing convergence quality and final fitness

---

## 🎮 Scenario Management

The project supports **multiple tactical scenarios**, allowing the optimizer to be tested under different spatial and contextual conditions.
A *scenario* defines the **initial conditions** of the optimization problem, including pitch configuration, ball position, obstacles, and team orientation.

Scenarios are managed through the `utils/setup.py` module and can be selected or customized from the command line.

### 🧩 What Is a Scenario?

A scenario encapsulates:

* Pitch dimensions and orientation
* Initial **ball position**
* Home and away team reference formations
* Optional **obstacles or spatial constraints**
* Phase-specific assumptions (offensive, defensive, non-possession)

This abstraction makes experiments **repeatable, comparable, and extensible**.

---

### ⚙️ Scenario Configuration

Scenarios are defined programmatically inside `setup_scenario()` and can be:

* **Default**: derived directly from tracking data averages
* **Custom**: manually configured for controlled experiments

Typical customizations include:

* Forcing the ball into specific zones (e.g., wide areas, central buildup)
* Reducing the effective pitch size to simulate compact phases
* Introducing obstacles or forbidden regions

---

### ▶️ Selecting a Scenario from CLI

The scenario can be selected at runtime using the `--scenario` argument:

```bash
python main.py --mode cma_dynamic --scenario compact_block --phase op
```

If no scenario is specified, the **default scenario** for the chosen phase is used.

---

## ▶️ Running the Project

### 1️⃣ Requirements

* **Python ≥ 3.9**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Command-Line Interface

The project supports configuration via CLI arguments.

#### Available Arguments

| Argument     | Description                     | Values                                     |
| ------------ | ------------------------------- | ------------------------------------------ |
| `--mode`     | Optimization algorithm          | `cma_static`, `cma_dynamic`, `de`          |
| `--scenario` | Tactical scenario configuration | optional                                   |
| `--phase`    | Phase of play                   | `op (Offensive possession)`, `dp (Defensive possession)`, `d (Defensive phase)` |

---

### 3️⃣ Example Commands

#### Static CMA-ES

```bash
python main.py --mode cma_static --phase op
```

#### Dynamic CMA-ES

```bash
python main.py --mode cma_dynamic --phase op
```

#### Differential Evolution

```bash
python main.py --mode de --phase op
```

---

## 📈 Outputs

Each run generates:

* Optimized formation plots (**PNG**)
* Vertical pitch visualizations
* Optimization **convergence curves**
* **GIF animations** of evolutionary progress

All results are stored in timestamped experiment folders.

---

## 🚀 Authors

Developed by **Sam Nejati**, **Daniele Notarangelo**, and **Jasnoor Singh**.
