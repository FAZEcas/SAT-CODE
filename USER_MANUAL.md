# SAT Solver - Quick Start Guide

## Overview
This tool compares different SAT solving algorithms (DPLL, Davis-Putnam, and Resolution) on CNF formulas.

## Basic Usage

```bash
python SAT_SOLVER.py
```

This runs all algorithms on the default input files and saves results to `results.txt`.

## Command Line Options

- `--input`: Specify input files
  ```bash
  python SAT_SOLVER.py --input formula1.txt formula2.txt
  ```

- `--output`: Specify output file
  ```bash
  python SAT_SOLVER.py --output my_results.txt
  ```

- `--timeout`: Set maximum execution time in seconds (default: 300)
  ```bash
  python SAT_SOLVER.py --timeout 60
  ```

- `--skip-resolution`: Skip resolution algorithm (helpful for large formulas)
- `--skip-dp`: Skip Davis-Putnam algorithm (helpful for large formulas)

## Input Format

CNF formulas should be formatted as follows:
- First line: `n_vars n_clauses`
- Subsequent lines: space-separated integers representing literals
  - Positive integers represent variables
  - Negative integers represent negated variables

Example:
```
5 7
1 2
-1 3
-2 4
-3 5
-4 -5
1 -3 -5
2 3 5
```

## Output Format

The results file contains:
- Performance metrics for each algorithm on each input file
- Average performance comparison across all files
- Execution time comparison matrix

## Algorithms

- **DPLL**: Implemented with four variable selection strategies:
  - Random
  - Jeroslow-Wang
  - Most frequent
  - Shortest clause

- **Davis-Putnam**: The original algorithm using resolution as inference rule

- **Resolution**: A complete inference procedure that derives new clauses

## Tips

- For large formulas, use `--skip-resolution` to avoid potential timeouts
- DPLL is generally the fastest algorithm, especially with the shortest_clause strategy