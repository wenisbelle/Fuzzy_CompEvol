# Fuzzy vs Fitness: Evolutionary Tuning for Multi-UAV Surveillance

## Motivation

Autonomous surveillance using multi-agent UAV (Unmanned Aerial Vehicle) systems requires each drone to continuously decide **where to go next** in order to keep the entire monitored area as up-to-date as possible. This is a non-trivial task allocation problem: a drone must balance visiting uncertain (stale) regions against the cost of traveling there, while also coordinating with peer drones to avoid redundant coverage.

This repository investigates and compares **two different task-allocation decision systems** for this problem, both tuned through a **Genetic Algorithm (GA)**:

1. **Analytical Fitness Function** — a compact, hand-designed formula that scores candidate waypoints based on map uncertainty and travel distance.
2. **Fuzzy Logic System** — a Mamdani-style fuzzy inference engine whose membership-function shapes and rule table are optimized by the GA.

The central question is: *can a GA-tuned fuzzy system outperform a GA-tuned analytical formula on the same surveillance task, and how does each scale with fleet size?*

All simulations are executed inside [GradySim](https://github.com/Project-GraDySim/GraDySim), a discrete-event network simulator for mobile agents, and experiments are conducted for fleets of 3, 5, 7, and 10 drones over a 2 000-second mission.

---

## The Surveillance Task

The simulated environment is a 2D grid map (up to 50 × 50 cells). Each cell carries an **uncertainty** value in [0, 1]:

- Uncertainty starts at 1 (completely unknown).
- When a drone flies over a cell, its uncertainty is reset to 0.
- Uncertainty grows back at a constant rate over time (the *vanishing map* model), reflecting that knowledge becomes stale.

Drones operate at a fixed altitude and sense the map using a downward-facing camera whose footprint depends on altitude and camera angle. They communicate via broadcast heartbeats and point-to-point messages within a 200 m radio range.

**Coordination protocol:** When two drones come within communication range they share their current map and jointly compute new destinations — one for each drone — so they spread out and reduce overlap. The drone with the higher ID performs this computation and sends the other drone its new waypoint.

**Objective (minimized by the GA):** Accumulated uncertainty summed over all drones and all time steps during the mission, normalized per drone.

---

## Decision Systems

### Greedy and ε-Greedy (Baselines)

Before the optimized systems, two simple deterministic baselines are included for comparison:

- **Greedy** — always sends the drone to a cell (or pair of cells) with the maximum current uncertainty on the map. Simple but prone to getting stuck in local repetitive patterns and ignoring travel cost.
- **ε-Greedy** — same as Greedy, but with probability ε = 0.1 the drone picks a *random* destination instead. The random exploration helps escape repetitive behavior at the cost of occasionally wasting flight time.

Both baselines are parameter-free and serve as lower-bound references.

---

### Analytical Fitness Function

The fitness function scores each candidate cell `(i, j)` for a single drone as:

```
fitness(i,j) = avg_uncertainty_on_trajectory(i,j) - distance / distance_norm
```

where `avg_uncertainty_on_trajectory` is the average uncertainty of all cells inside the camera footprint along the straight-line path from the drone's current position to cell `(i, j)`, and `distance_norm` is a scaling parameter.

For the **two-drone coordination** case (on encounter), the combined score of assigning cell A to drone 1 and cell B to drone 2 is:

```
total_fitness = priority(A) + priority(B) + distance(A, B) / distance_between_drone_norm
```

The third term encourages spreading the two drones apart.

**GA-tuned parameters:** `distance_norm` and `distance_between_drone_norm` (2 real-valued genes). The GA minimizes accumulated uncertainty by finding the normalization scales that best balance exploration against travel efficiency.

---

### Fuzzy Logic System

The fuzzy approach replaces the explicit formula with two Mamdani fuzzy inference systems (FIS) built with `scikit-fuzzy`:

**System 1 — Single-drone cell scoring**

| Component | Variable | Linguistic sets |
|-----------|----------|-----------------|
| Input | Uncertainty of candidate cell | very_low, low, medium, high, very_high |
| Input | Distance to candidate cell | very_close, close, medium, far, very_far |
| Output | Cell priority | very_low, low, medium, high, very_high |

5 × 5 = **25 rules**, one per input-set combination.

**System 2 — Two-drone pair scoring**

| Component | Variable | Linguistic sets |
|-----------|----------|-----------------|
| Input | Sum of individual priorities for the pair | very_low, low, medium, high, very_high |
| Input | Distance between the two target cells | very_close, close, medium, far, very_far |
| Output | Pair priority | very_low, low, medium, high, very_high |

5 × 5 = **25 rules**, one per input-set combination.

Each membership function is parameterized by breakpoint intervals (5 intervals per variable, 6 variables = 30 genes). Each rule consequent is an integer in {0, 1, 2, 3, 4} encoding which output set fires (25 + 25 = 50 genes). Total genome size: **80 genes** (30 real-valued MF parameters + 50 integer rule genes).

To avoid re-computing defuzzification at runtime, both systems are pre-compiled into **RegularGridInterpolator lookup tables** (`scipy`), making inference fast enough for real-time-speed simulation.

**GA-tuned parameters:** all 80 genes simultaneously. The objective is identical to the fitness function case (minimize accumulated uncertainty). Each individual is evaluated by averaging 3 independent simulation runs to reduce stochasticity.

---



## Installation

### System Requirements

- [Docker Engine](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(optional — only needed for GPU-accelerated workloads; remove the `deploy` section from `docker-compose.yml` to run CPU-only)*

### Build and Start

Clone the repository:

```bash
git clone https://github.com/wenisbelle/Fuzzy_CompEvol.git
cd Fuzzy_CompEvol
```

Allow the container to open graphical windows (needed for the visualizer):

```bash
xhost local:+
```

Build the Docker image:

```bash
docker build -t fuzzy_ga .
```

Start the container:

```bash
docker compose up -d
```

Open an interactive shell inside the container:

```bash
docker exec -it fuzzy_ga_container bash
```

---

## How to Use

### Running a Simulation (Analytical Fitness)

```bash
cd /FuzzyGA/src
python -m 3_drones.analytical.test.execution
```

The script runs 20 evaluation episodes with the best-known individual `[3611.5, 3563.1]` and writes results to the log file under `logs/`.

### Running the GA Tuning (Fuzzy System)

Navigate to the relevant `tune/` directory and run `execution.py`. The GA uses DEAP with a `(μ + λ)` strategy, evaluates each individual over 3 stochastic simulation runs, and writes a `ga_logbook.txt` convergence file.

```bash
python -m definitive_system.coordination_only.3_drones.tune_more_rules.tune.execution
```

### Plotting GA Convergence

Compare convergence curves from up to 4 logbook files:

```bash
python src/plot_ga_convergence_multi.py \
    log1.txt log2.txt log3.txt log4.txt \
    --labels "Fuzzy 3dr" "Fuzzy 5dr" "Analytical 3dr" "Analytical 5dr" \
    --max-gen 20 \
    --output convergence_multi.png
```

Arguments:

| Flag | Description |
|------|-------------|
| `logbooks` (positional) | Paths to exactly 4 GA logbook `.txt` files |
| `--labels` / `-l` | Legend labels (4 values); defaults to filenames |
| `--max-gen` / `-g` | Truncate plot at this generation (default: 20) |
| `--output` / `-o` | Save path for the PNG; shows interactively if omitted |

