# SAT Solver Comparison Tool

This repository contains a Python implementation of various SAT (Boolean Satisfiability Problem) solving algorithms with different variable selection heuristics. The main purpose is to compare the performance of different algorithms and strategies on CNF (Conjunctive Normal Form) formulas.

## Implemented Algorithms

- **DPLL (Davis-Putnam-Logemann-Loveland)**: A complete, backtracking-based search algorithm for deciding the satisfiability of propositional logic formulas in conjunctive normal form. Implemented with several variable selection heuristics:
  - Random selection
  - Jeroslow-Wang heuristic
  - Most frequent variable
  - Shortest clause first

- **Davis-Putnam Algorithm**: The original Davis-Putnam algorithm that uses resolution as its main inference rule.

- **Resolution**: A proof by contradiction method that attempts to refute the formula by deriving a contradiction.

## Requirements

- Python 3.6+

## Usage

```bash
python SAT_SOLVER.py [options]
```

### Command-line Options

- `--input`: List of input CNF files to analyze (default: input_1.txt, input_2.txt, input_3.txt)
- `--output`: Output file for results (default: results.txt)
- `--timeout`: Timeout in seconds for each algorithm (default: 300)
- `--skip-resolution`: Skip the resolution algorithm (useful for complex formulas)
- `--skip-dp`: Skip the Davis-Putnam algorithm (useful for complex formulas)

### Example

```bash
python SAT_SOLVER.py --input my_formula.txt --output my_results.txt --timeout 60
```

## Input Format

Input files should contain CNF formulas in the following format:
- The first line should contain two integers: the number of variables and the number of clauses.
- Each subsequent line represents a clause, with integers representing literals (positive for variables, negative for negated variables).

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

## Output

The program will generate a results file containing performance statistics for each algorithm and strategy including:
- Execution time
- Memory usage
- Number of decisions (for DPLL)
- Number of unit propagations (for DPLL)
- Result (SAT/UNSAT)
