#!/usr/bin/env python3
"""
Solve a single Cube2 scramble and print the solution moves and elapsed time.

Usage examples:
  1) Random scramble of length 10 (default model path):
     python scripts/solve_one_cube2.py --model_dir saved_models/cube2/current --scramble_len 10

  2) Provide an explicit scramble move list (applied from goal using next_state):
     python scripts/solve_one_cube2.py --model_dir saved_models/cube2/current \
       --scramble_moves "U1 R-1 F1 U-1"

  3) Provide a raw state (24 integers, space-separated or comma-separated):
     python scripts/solve_one_cube2.py --model_dir saved_models/cube2/current \
       --state_colors "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"

Notes:
 - Requires a trained model at <model_dir>/model_state_dict.pt
 - Uses the existing A* implementation with neural heuristic.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

from environments.cube2 import Cube2, Cube2State
from utils import nnet_utils
from search_methods.astar import AStar, get_path


def parse_moves(env: Cube2, moves_str: str) -> List[int]:
    """Parse a space-separated move string into env action indices.
    Accepts tokens like 'U1', 'U-1', 'R1', etc.
    """
    if not moves_str:
        return []
    tokens = [t.strip() for t in moves_str.replace(",", " ").split() if t.strip()]
    move_to_idx = {m: i for i, m in enumerate(env.moves)}
    try:
        return [move_to_idx[t] for t in tokens]
    except KeyError as e:
        valid = " ".join(env.moves[:6]) + " ..."
        raise ValueError(f"Unknown move '{e.args[0]}'. Example valid moves: {valid}")


def parse_state_colors(state_str: str) -> np.ndarray:
    """Parse 24 integers into a colors array for Cube2State."""
    # allow comma or space
    tokens = [t for t in state_str.replace(",", " ").split() if t]
    if len(tokens) != 24:
        raise ValueError(f"state_colors must have exactly 24 integers, got {len(tokens)}")
    try:
        arr = np.array([int(t) for t in tokens], dtype=np.uint8)
    except ValueError:
        raise ValueError("state_colors must be integers (0..23)")
    return arr


def scramble_random(env: Cube2, length: int) -> Tuple[Cube2State, List[int]]:
    """Generate one random scrambled state and return both state and applied move indices.

    We build the scramble explicitly so we can print the exact sequence.
    """
    state = env.generate_goal_states(1)[0]
    moves_idx: List[int] = []
    if length <= 0:
        return state, moves_idx
    rng = np.random.default_rng()
    for _ in range(int(length)):
        m = int(rng.integers(0, env.get_num_moves()))
        state = env.next_state([state], m)[0][0]
        moves_idx.append(m)
    return state, moves_idx


def scramble_with_moves(env: Cube2, moves_idx: List[int]) -> Cube2State:
    """Start from goal and apply next_state with the given move indices to get a scrambled state."""
    state = env.generate_goal_states(1)[0]
    for m in moves_idx:
        state = env.next_state([state], m)[0][0]
    return state


def solve_one(state: Cube2State, env: Cube2, model_dir: str, weight: float,
              batch_size: int, nnet_batch_size: int | None, verbose: bool) -> Tuple[List[int], float, int, dict]:
    device, devices, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn(model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=True, batch_size=nnet_batch_size)

    start_time = time.time()
    astar = AStar([state], env, heuristic_fn, [weight])
    while not min(astar.has_found_goal()):
        astar.step(heuristic_fn, batch_size, verbose=verbose)
    goal_node = astar.get_goal_node_smallest_path_cost(0)
    path, soln, path_cost = get_path(goal_node)
    elapsed = time.time() - start_time
    nodes_generated = astar.get_num_nodes_generated(0)
    timings = dict(astar.timings)
    return soln, elapsed, nodes_generated, timings


def main():
    parser = argparse.ArgumentParser(description="Solve a single Cube2 scramble with A*+NN heuristic")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing model_state_dict.pt for Cube2")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight w in f = w*g + h")
    parser.add_argument("--batch_size", type=int, default=64, help="A* expansion batch size")
    parser.add_argument("--nnet_batch_size", type=int, default=None,
                        help="Heuristic NN batch size per GPU (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration stats")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scramble_len", type=int, default=10, help="Random scramble length (if no moves/state)")
    group.add_argument("--scramble_moves", type=str, default=None,
                       help="Space/comma separated moves, e.g., 'U1 R-1 F1'. Applied from goal with next_state.")
    group.add_argument("--state_colors", type=str, default=None,
                       help="Raw Cube2 state (24 integers). Overrides scramble options if provided.")

    args = parser.parse_args()

    model_file = os.path.join(args.model_dir, "model_state_dict.pt")
    if not os.path.isfile(model_file):
        print(f"Model not found: {model_file}", file=sys.stderr)
        sys.exit(2)

    env = Cube2()

    # Build the start state
    scramble_moves_idx: List[int] | None = None

    if args.state_colors:
        colors = parse_state_colors(args.state_colors)
        state = Cube2State(colors)
        scramble_desc = "provided state_colors"
    elif args.scramble_moves:
        try:
            moves_idx = parse_moves(env, args.scramble_moves)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        state = scramble_with_moves(env, moves_idx)
        scramble_moves_idx = moves_idx
        scramble_desc = f"moves: {args.scramble_moves}"
    else:
        state, scramble_moves_idx = scramble_random(env, max(0, int(args.scramble_len)))
        scramble_desc = f"random length {args.scramble_len}"

    # Solve
    soln, elapsed, nodes_generated, timings = solve_one(state, env, args.model_dir, args.weight,
                                                        args.batch_size, args.nnet_batch_size, args.verbose)

    # Pretty-print
    moves_human = [env.moves[m] for m in soln]
    if scramble_moves_idx is not None and len(scramble_moves_idx) > 0:
        scramble_moves_human = [env.moves[m] for m in scramble_moves_idx]
        print(f"Scramble moves (idx): {scramble_moves_idx}")
        print(f"Scramble moves (human): {' '.join(scramble_moves_human)}")
    else:
        print("Scramble moves: [not available for provided raw state]")
    print("=== Cube2 single solve ===")
    print(f"Scramble: {scramble_desc}")
    print(f"Solution length: {len(soln)}")
    print(f"Solution moves (idx): {soln}")
    print(f"Solution moves (human): {' '.join(moves_human)}")
    print(f"Elapsed time (s): {elapsed:.4f}")
    print(f"Nodes generated: {nodes_generated}")
    print("Timings:", ", ".join([f"{k}:{v:.3f}" for k, v in timings.items()]))


if __name__ == "__main__":
    main()
