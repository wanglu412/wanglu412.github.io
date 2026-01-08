"""
Script to run a command multiple times
Usage: python run_multiple.py <num_times> <command>
Example: python run_multiple.py 5 "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"
"""

import sys
import subprocess
import time
import json
import os
import re
import numpy as np
from datetime import datetime


def extract_test_auc_from_results(results_file='dann_results.json'):
    """
    Extract test AUC from results file

    Args:
        results_file: Path to results JSON file

    Returns:
        test_auc value or None if not found
    """
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                # Extract test_auc from results
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0].get('test_auc', None)
        return None
    except Exception as e:
        print(f"Warning: Could not extract test AUC from {results_file}: {e}")
        return None


def save_statistics_report(dataset_name, test_aucs, num_times, output_file='statistics_report.json'):
    """
    Save statistics report with mean and std of test AUC

    Args:
        dataset_name: Name of the dataset
        test_aucs: List of test AUC values
        num_times: Number of runs
        output_file: Output file path
    """
    if len(test_aucs) == 0:
        print(f"Warning: No test AUC values collected, cannot generate statistics")
        return

    mean_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs, ddof=1) if len(test_aucs) > 1 else 0.0

    report = {
        'dataset': dataset_name,
        'num_runs': num_times,
        'successful_runs': len(test_aucs),
        'test_auc_values': test_aucs,
        'statistics': {
            'mean': float(mean_auc),
            'std': float(std_auc),
            'min': float(np.min(test_aucs)),
            'max': float(np.max(test_aucs))
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nStatistics report saved to: {output_file}")

    return report


def run_command_multiple_times(command, num_times=5, results_file='dann_results.json'):
    """
    Run a command multiple times and track results

    Args:
        command: Command to run (as string or list)
        num_times: Number of times to run the command
        results_file: Path to results file to extract test AUC from
    """
    success_count = 0
    fail_count = 0
    results = []
    test_aucs = []  # Collect test AUC values

    # Try to extract dataset name from command
    dataset_name = "unknown"
    dataset_match = re.search(r'--dataset\s+(\S+)', command)
    if dataset_match:
        dataset_name = dataset_match.group(1)

    print("=" * 80)
    print(f"Running command {num_times} times")
    print(f"Command: {command}")
    print(f"Dataset: {dataset_name}")
    print("=" * 80)
    print()

    for i in range(1, num_times + 1):
        print()
        print("=" * 80)
        print(f"Run {i} / {num_times}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        start_time = time.time()

        # Run command
        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True)
            else:
                result = subprocess.run(command)

            exit_code = result.returncode
            elapsed_time = time.time() - start_time

            if exit_code == 0:
                success_count += 1
                status = "SUCCESS"
                print()
                print(f"[Run {i}] SUCCESS (took {elapsed_time:.2f} seconds)")

                # Extract test AUC from results file
                test_auc = extract_test_auc_from_results(results_file)
                if test_auc is not None:
                    test_aucs.append(test_auc)
                    print(f"[Run {i}] Test AUC: {test_auc:.4f}")
                else:
                    print(f"[Run {i}] Warning: Could not extract test AUC")

            else:
                fail_count += 1
                status = "FAILED"
                print()
                print(f"[Run {i}] FAILED with error code {exit_code} (took {elapsed_time:.2f} seconds)")

            results.append({
                'run': i,
                'status': status,
                'exit_code': exit_code,
                'time': elapsed_time,
                'test_auc': test_aucs[-1] if test_aucs and exit_code == 0 else None
            })

        except Exception as e:
            fail_count += 1
            elapsed_time = time.time() - start_time
            print()
            print(f"[Run {i}] FAILED with exception: {str(e)}")
            results.append({
                'run': i,
                'status': 'FAILED',
                'exit_code': -1,
                'time': elapsed_time,
                'error': str(e),
                'test_auc': None
            })

        print()
        if i < num_times:
            print("Continuing to next run...")
            time.sleep(3)

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total runs: {num_times}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    print("Detailed results:")
    for result in results:
        status_symbol = "✓" if result['status'] == "SUCCESS" else "✗"
        auc_str = f", AUC: {result['test_auc']:.4f}" if result['test_auc'] is not None else ""
        print(f"  Run {result['run']}: {status_symbol} {result['status']} "
              f"(exit code: {result['exit_code']}, time: {result['time']:.2f}s{auc_str})")
    print("=" * 80)

    # Calculate and display statistics
    if len(test_aucs) > 0:
        mean_auc = np.mean(test_aucs)
        std_auc = np.std(test_aucs, ddof=1) if len(test_aucs) > 1 else 0.0
        min_auc = np.min(test_aucs)
        max_auc = np.max(test_aucs)

        print()
        print("=" * 80)
        print("TEST AUC STATISTICS")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Successful runs with AUC: {len(test_aucs)}")
        print()
        print(f"  Mean AUC:  {mean_auc:.4f}")
        print(f"  Std AUC:   {std_auc:.4f}")
        print(f"  Min AUC:   {min_auc:.4f}")
        print(f"  Max AUC:   {max_auc:.4f}")
        print()
        print(f"  Result: {mean_auc:.4f} ± {std_auc:.4f}")
        print("=" * 80)

        # Save statistics report
        stats_file = f'statistics_{dataset_name}.json'
        save_statistics_report(dataset_name, test_aucs, num_times, stats_file)
    else:
        print()
        print("=" * 80)
        print("WARNING: No test AUC values collected!")
        print("=" * 80)

    return success_count, fail_count, test_aucs


def main():
    if len(sys.argv) < 2:
        print("Error: No command provided")
        print()
        print("Usage: python run_multiple.py <num_times> <command>")
        print("   or: python run_multiple.py <command>  (default: 5 times)")
        print()
        print("Examples:")
        print('  python run_multiple.py 5 "python train_dann.py --dataset goodhiv_scaffold_covariate"')
        print('  python run_multiple.py "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"')
        sys.exit(1)

    # Parse arguments
    try:
        num_times = int(sys.argv[1])
        command = " ".join(sys.argv[2:])
    except ValueError:
        # If first argument is not a number, use default 5 times
        num_times = 5
        command = " ".join(sys.argv[1:])

    if not command:
        print("Error: No command provided")
        sys.exit(1)

    # Run command
    success_count, fail_count, test_aucs = run_command_multiple_times(command, num_times)

    # Exit with error if any runs failed
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
