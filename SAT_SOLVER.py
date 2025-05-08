import time
import os
import random
import sys
import argparse
import tracemalloc
from collections import defaultdict
import signal

sys.setrecursionlimit(10000)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Algorithm timed out!")


class SATSolver:
    def __init__(self, strategy="random"):
        self.strategy = strategy
        self.stats = {}

    def parse_cnf(self, filename):
        clauses = []
        n_vars = 0

        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            n_vars, n_clauses = map(int, first_line.split())

            for line in f:
                line = line.strip()
                if line:
                    clause = [int(x) for x in line.split() if x != '0']
                    if clause:
                        clauses.append(clause)

        return clauses, n_vars

    def _choose_random_variable(self, unassigned):
        return random.choice(unassigned) if unassigned else None

    def _choose_jeroslow_wang_variable(self, formula, unassigned):
        j_scores = defaultdict(float)
        for clause in formula:
            for lit in clause:
                var = abs(lit)
                if var in unassigned:
                    j_scores[var] += 2 ** -len(clause)
        return max(j_scores.items(), key=lambda x: x[1])[0] if j_scores else self._choose_random_variable(unassigned)

    def _choose_most_frequent_variable(self, formula, unassigned):
        freq = defaultdict(int)
        for clause in formula:
            for lit in clause:
                var = abs(lit)
                if var in unassigned:
                    freq[var] += 1
        return max(freq.items(), key=lambda x: x[1])[0] if freq else self._choose_random_variable(unassigned)

    def _choose_shortest_clause_variable(self, formula, unassigned):
        min_len = float('inf')
        candidates = []
        for clause in formula:
            unassigned_lits = [lit for lit in clause if abs(lit) in unassigned]
            if unassigned_lits and len(unassigned_lits) < min_len:
                min_len = len(unassigned_lits)
                candidates = unassigned_lits
        return abs(random.choice(candidates)) if candidates else self._choose_random_variable(unassigned)

    def choose_variable(self, formula, assignment, n_vars):
        unassigned = [i for i in range(1, n_vars + 1) if i not in assignment and -i not in assignment]
        if not unassigned:
            return None

        if self.strategy == "random":
            return self._choose_random_variable(unassigned)
        elif self.strategy == "jeroslow_wang":
            return self._choose_jeroslow_wang_variable(formula, unassigned)
        elif self.strategy == "most_frequent":
            return self._choose_most_frequent_variable(formula, unassigned)
        elif self.strategy == "shortest_clause":
            return self._choose_shortest_clause_variable(formula, unassigned)
        return self._choose_random_variable(unassigned)

    def unit_propagation(self, formula, assignment):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(formula):
                clause = formula[i]

                # If clause is satisfied, remove it
                if any(lit in assignment for lit in clause):
                    formula.pop(i)
                    changed = True
                    continue

                # Get literals not assigned false
                unassigned = [lit for lit in clause if -lit not in assignment]

                # Conflict found
                if not unassigned:
                    self.stats["conflicts"] += 1
                    return formula, assignment, False

                # Unit clause found
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    formula.pop(i)

                    # Check for contradiction
                    if -lit in assignment:
                        self.stats["conflicts"] += 1
                        return formula, assignment, False

                    if lit not in assignment:
                        assignment.append(lit)
                        self.stats["unit_propagations"] += 1
                        changed = True
                    continue

                i += 1

        return formula, assignment, True

    def dpll(self, formula, assignment, n_vars):
        formula, assignment, status = self.unit_propagation(formula, assignment)
        if not status:
            self.stats["backtracks"] += 1
            return False

        if not formula:
            return assignment

        var = self.choose_variable(formula, assignment, n_vars)
        if var is None:
            return assignment if not formula else False

        self.stats["decisions"] += 1

        new_assignment = assignment + [var]
        result = self.dpll(formula.copy(), new_assignment, n_vars)
        if result:
            return result

        new_assignment = assignment + [-var]
        return self.dpll(formula.copy(), new_assignment, n_vars)

    def _measure_performance(self, func, filename):
        tracemalloc.start()
        start_time = time.time()
        result = func(filename)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        memory_used = peak / 1024 / 1024
        tracemalloc.stop()
        stats = {
            "time": end_time - start_time,
            "memory": memory_used,
        }
        if isinstance(result, dict) and 'stats' in result:
            stats.update(result['stats'])
            solver_result = result.get('result')
        else:
            solver_result = result
        return stats, solver_result

    def solve_dpll(self, filename):
        clauses, n_vars = self.parse_cnf(filename)
        self.stats = {"decisions": 0, "unit_propagations": 0, "backtracks": 0, "conflicts": 0}
        stats, result = self._measure_performance(
            lambda f: {"result": self.dpll(clauses, [], n_vars), "stats": self.stats}, filename)
        return {**stats, "result": "SAT" if result else "UNSAT"}

    def resolve(self, clause1, clause2, var):
        result = []
        for lit in clause1:
            if lit != var:
                result.append(lit)
        for lit in clause2:
            if lit != -var and lit not in result:
                result.append(lit)
        return result

    def resolution(self, clauses):
        while True:
            # Track if we found resolvable clauses
            found_resolvable = False
            new_clauses = []

            # Look for clauses that can be resolved
            for i, clause1 in enumerate(clauses):
                for j, clause2 in enumerate(clauses[i + 1:], i + 1):
                    # Check if clauses contain complementary literals
                    for lit in clause1:
                        if -lit in clause2:
                            found_resolvable = True
                            resolvent = self.resolve(clause1, clause2, lit)

                            # Skip tautologies
                            if any(l in resolvent and -l in resolvent for l in resolvent):
                                continue

                            # Add new unique clause
                            if resolvent and resolvent not in clauses and resolvent not in new_clauses:
                                new_clauses.append(resolvent)

                            # Empty clause means UNSAT
                            if not resolvent:
                                return False

            # If no resolutions found or no new clauses added, formula is SAT
            if not found_resolvable or not new_clauses:
                return True

            # Add new clauses and continue
            clauses.extend(new_clauses)

    def solve_resolution(self, filename):
        clauses, _ = self.parse_cnf(filename)
        stats, result = self._measure_performance(lambda f: self.resolution(clauses), filename)
        return {**stats, "result": "SAT" if result else "UNSAT"}

    def davis_putnam_original(self, clauses, n_vars):
        if not clauses:
            return True
        if any(not clause for clause in clauses):
            return False

        unit_clauses = [clause[0] for clause in clauses if len(clause) == 1]
        if unit_clauses:
            lit = unit_clauses[0]
            new_clauses = []
            for clause in clauses:
                if lit in clause:
                    continue
                new_clause = [l for l in clause if l != -lit]
                if not new_clause:
                    return False
                new_clauses.append(new_clause)
            return self.davis_putnam_original(new_clauses, n_vars)

        for var in range(1, n_vars + 1):
            if any(var in clause or -var in clause for clause in clauses):
                break
        else:
            return True

        pos_clauses = []
        neg_clauses = []
        other_clauses = []

        for clause in clauses:
            if var in clause:
                pos_clauses.append([lit for lit in clause if lit != var])
            elif -var in clause:
                neg_clauses.append([lit for lit in clause if lit != -var])
            else:
                other_clauses.append(clause)

        resolvents = []
        for pos_clause in pos_clauses:
            for neg_clause in neg_clauses:
                resolvent = list(set(pos_clause + neg_clause))
                if not any(lit in resolvent and -lit in resolvent for lit in resolvent) and resolvent not in resolvents:
                    resolvents.append(resolvent)

        new_clauses = other_clauses + resolvents
        return self.davis_putnam_original(new_clauses, n_vars)

    def solve_dp_original(self, filename):
        clauses, n_vars = self.parse_cnf(filename)
        stats, result = self._measure_performance(lambda f: self.davis_putnam_original(clauses, n_vars), filename)
        return {**stats, "result": "SAT" if result else "UNSAT"}


def write_results(results, output_file="results.txt"):
    with open(output_file, 'w') as f:
        f.write("SAT Solver Performance Summary\n\n")

        input_files = sorted(set(result['input'] for result in results))
        organized_results = {}

        for result in results:
            algorithm = result['algorithm']
            strategy = result.get('strategy', 'N/A')
            key = f"{algorithm} ({strategy})" if strategy != 'N/A' else algorithm
            input_file = result['input']

            if key not in organized_results:
                organized_results[key] = {
                    'by_input': {},
                    'total_time': 0,
                    'total_memory': 0,
                    'total_decisions': 0,
                    'total_propagations': 0,
                    'count': 0
                }

            time_taken = result['stats']['time']
            memory_used = result['stats']['memory']
            decisions = result['stats'].get('decisions', 0)
            unit_propagations = result['stats'].get('unit_propagations', 0)
            sat_result = result['stats'].get('result', 'N/A')

            organized_results[key]['by_input'][input_file] = {
                'time': time_taken,
                'memory': memory_used,
                'decisions': decisions,
                'unit_propagations': unit_propagations,
                'result': sat_result
            }

            organized_results[key]['total_time'] += time_taken
            organized_results[key]['total_memory'] += memory_used
            organized_results[key]['total_decisions'] += decisions
            organized_results[key]['total_propagations'] += unit_propagations
            organized_results[key]['count'] += 1

        sorted_algos = sorted(organized_results.keys())

        for input_file in input_files:
            f.write(f"\nResults for {input_file}\n")
            header = f"{'Algorithm':<25} {'Time (s)':<10} {'Memory':<10} {'Decisions':<10} {'Unit Props':<10} {'Result':<6}"
            f.write(header + "\n")

            input_sorted_algos = sorted(
                [algo for algo in sorted_algos if input_file in organized_results[algo]['by_input']],
                key=lambda x: organized_results[x]['by_input'][input_file]['time']
            )

            for algo in input_sorted_algos:
                data = organized_results[algo]['by_input'].get(input_file)
                if data:
                    line = f"{algo:<25} {data['time']:<10.4f} {data['memory']:<10.2f} {data['decisions']:<10} {data['unit_propagations']:<10} {data['result']:<6}"
                    f.write(line + "\n")

        f.write("\nAverage Performance Comparison\n")
        header = f"{'Algorithm':<25} {'Avg Time':<10} {'Avg Memory':<10} {'Avg Decisions':<12} {'Avg Unit Props':<12}"
        f.write(header + "\n")

        avg_sorted_algos = sorted(
            sorted_algos,
            key=lambda x: organized_results[x]['total_time'] / organized_results[x]['count']
        )

        for algo in avg_sorted_algos:
            data = organized_results[algo]
            avg_time = data['total_time'] / data['count']
            avg_memory = data['total_memory'] / data['count']
            avg_decisions = data['total_decisions'] / data['count']
            avg_propagations = data['total_propagations'] / data['count']

            line = f"{algo:<25} {avg_time:<10.4f} {avg_memory:<10.2f} {avg_decisions:<12.2f} {avg_propagations:<12.2f}"
            f.write(line + "\n")

        f.write("\nAlgorithm Performance Across All Inputs\n")
        header = f"{'Algorithm':<25}"
        for input_file in input_files:
            header += f"{input_file:<12}"
        f.write(header + "\n")

        for algo in avg_sorted_algos:
            line = f"{algo:<25}"
            for input_file in input_files:
                data = organized_results[algo]['by_input'].get(input_file)
                if data:
                    line += f"{data['time']:<12.4f}"
                else:
                    line += f"{'N/A':<12}"
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description='SAT Solver Comparison')
    parser.add_argument('--input', nargs='+', default=['input_1.txt', 'input_2.txt', 'input_3.txt'],
                        help='Input CNF files to analyze')
    parser.add_argument('--output', default='results.txt', help='Output file for results')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each algorithm')
    parser.add_argument('--skip-resolution', action='store_true',
                        help='Skip resolution algorithm (may be slow on complex formulas)')
    parser.add_argument('--skip-dp', action='store_true',
                        help='Skip Davis-Putnam algorithm (may be slow on complex formulas)')

    args = parser.parse_args()

    for input_file in args.input:
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist.")
            sys.exit(1)

    if args.timeout > 0:
        import signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)
        try:
            results = []
            dpll_strategies = ["random", "jeroslow_wang", "most_frequent", "shortest_clause"]

            for input_file in args.input:
                print(f"Processing {input_file}...")

                for strategy in dpll_strategies:
                    print("Running DPLL")
                    solver = SATSolver(strategy=strategy)
                    try:
                        stats = solver.solve_dpll(input_file)
                        results.append({
                            'algorithm': 'DPLL',
                            'strategy': strategy,
                            'input': input_file,
                            'stats': stats
                        })
                        print("Completed")
                    except TimeoutError as e:
                        print("Timeout")
                    except Exception as e:
                        print("Failed")

                if not args.skip_dp:
                    print("Running Davis-Putnam")
                    solver = SATSolver()
                    try:
                        stats = solver.solve_dp_original(input_file)
                        results.append({
                            'algorithm': 'Davis-Putnam',
                            'input': input_file,
                            'stats': stats
                        })
                        print("Completed")
                    except TimeoutError as e:
                        print("Timeout")
                    except Exception as e:
                        print("Failed")

                if not args.skip_resolution:
                    print("Running Resolution")
                    solver = SATSolver()
                    try:
                        stats = solver.solve_resolution(input_file)
                        results.append({
                            'algorithm': 'Resolution',
                            'input': input_file,
                            'stats': stats
                        })
                        print("Completed")
                    except TimeoutError as e:
                        print("Timeout")
                    except Exception as e:
                        print("Failed")

            write_results(results, args.output)
            print(f"\nResults written to {args.output}")

        finally:
            if args.timeout > 0:
                signal.alarm(0)

    else:
        results = []
        dpll_strategies = ["random", "jeroslow_wang", "most_frequent", "shortest_clause"]

        for input_file in args.input:
            print(f"Processing {input_file}...")

            for strategy in dpll_strategies:
                print("Running DPLL")
                solver = SATSolver(strategy=strategy)
                try:
                    stats = solver.solve_dpll(input_file)
                    results.append({
                        'algorithm': 'DPLL',
                        'strategy': strategy,
                        'input': input_file,
                        'stats': stats
                    })
                    print("Completed")
                except Exception as e:
                    print("Failed")

            if not args.skip_dp:
                print("Running Davis-Putnam")
                solver = SATSolver()
                try:
                    stats = solver.solve_dp_original(input_file)
                    results.append({
                        'algorithm': 'Davis-Putnam',
                        'input': input_file,
                        'stats': stats
                    })
                    print("Completed")
                except Exception as e:
                    print("Failed")

            if not args.skip_resolution:
                print("Running Resolution")
                solver = SATSolver()
                try:
                    stats = solver.solve_resolution(input_file)
                    results.append({
                        'algorithm': 'Resolution',
                        'input': input_file,
                        'stats': stats
                    })
                    print("Completed")
                except Exception as e:
                    print("Failed")

        write_results(results, args.output)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()