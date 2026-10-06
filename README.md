# Introduction to AI (CS 440) - Fire Escape Simulation

This repository contains a simulation environment for Project 1 of CS 440 at Rutgers: Introduction to Artificial Intelligence, the goal is navigating an agent through a dynamic and hazardous environment. The scenario involves a bot on a spaceship where a fire has started. The bot's goal is to reach an emergency button before the spreading fire engulfs it or its path.

The project explores and compares several pathfinding strategies, from simple reactive planning to complex proactive planning that accounts for the predicted spread of the fire.

## Project Structure

*   `ship.py`: Generates the spaceship's grid layout, creating a connected maze of open and blocked cells.
*   `fire.py`: Manages the fire simulation, spreading it probabilistically at each time step based on a flammability quotient `q`.
*   `bots.py`: Defines the different AI agents (bots) with distinct pathfinding and decision-making strategies.
*   `pathfinding.py`: Implements the core pathfinding algorithms used by the bots, including Breadth-First Search (BFS), A*, and a specialized A* that navigates through space and time while considering fire risk.
*   `simulation.py`: Contains the engine for running a single trial, pitting one bot against the fire on a specific ship layout.
*   `experiments.py`: A script for running reproducible experiments to compare the performance of all bots across a range of scenarios and fire spread probabilities.

## Bot Strategies

This project implements four bots, each with an increasingly sophisticated strategy for navigating the burning ship.

*   **Bot 1**: Plans its path once at the start of the simulation using A*, avoiding only the initial fire cell. It follows this pre-determined path without deviation, regardless of how the fire spreads.

*   **Bot 2**: Re-plans its path at every time step using A*. This strategy is reactive, calculating the shortest path to the goal while avoiding all cells that are currently on fire.

*   **Bot 3**: An improvement on Bot 2. It re-plans at every step and attempts to find a path that avoids both burning cells and any open cells adjacent to the fire. If such a "cautious" path is not possible, it reverts to Bot 2's behavior of just avoiding the burning cells.

*   **Bot 4**: The most advanced agent. This bot is proactive, using a probabilistic model to predict the risk of fire spreading to each cell over a future time horizon. It then uses a specialized A* algorithm to search for the safest path through space and time, balancing path length with the cumulative risk of encountering fire. It replans its path periodically to adapt to the actual fire spread.

## How to Run Experiments

The primary entry point for this project is `experiments.py`, which runs a sweep of simulations to compare the bots' performance under various conditions.

To run the default experiment suite:
```bash
python Project1/experiments.py
```
This will run each bot over a series of trials for different fire flammability (`q`) values and output the results to the console and a CSV file.

### Command-Line Arguments

You can customize the simulation parameters using the following arguments:

*   `--grid-size`: The dimension `D` of the square ship grid. Default: `75`.
*   `--trials`: The number of trials to run for each `q` value. Default: `10`.
*   `--seed`: The master random seed for generating reproducible scenarios. Default: `440`.
*   `--max-steps`: The maximum number of steps in a single trial before a timeout. Default: `100000`.
*   `--csv`: The path to the output CSV file for storing experiment results. Default: `bot4_sweep.csv`.

Example:
```bash
python Project1/experiments.py --grid-size 50 --trials 100 --csv results.csv
