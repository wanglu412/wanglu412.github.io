"""
Script to train DANN on multiple datasets SEQUENTIALLY (one by one, not mixed)

IMPORTANT: This script trains each dataset SEPARATELY, one after another.
           It does NOT mix datasets in a single training run.
           Each dataset gets its own model and evaluation.

Usage:
    python train_all_datasets.py good    # Train all 12 GOOD configurations
    python train_all_datasets.py ic50    # Train all 3 IC50 configurations
    python train_all_datasets.py ec50    # Train all 3 EC50 configurations
    python train_all_datasets.py all     # Train all 18 configurations
"""

import subprocess
import sys

# All GOOD dataset configurations
good_datasets = [
    # GOODHIV
    'goodhiv_scaffold_covariate',
    'goodhiv_scaffold_concept',
    'goodhiv_size_covariate',
    'goodhiv_size_concept',

    # GOODZINC
    'goodzinc_scaffold_covariate',
    'goodzinc_scaffold_concept',
    'goodzinc_size_covariate',
    'goodzinc_size_concept',

    # GOODPCBA
    'goodpcba_scaffold_covariate',
    'goodpcba_scaffold_concept',
    'goodpcba_size_covariate',
    'goodpcba_size_concept',
]

# DrugOOD IC50 configurations
ic50_datasets = [
    'ic50_assay',
    'ic50_scaffold',
    'ic50_size',
]

# DrugOOD EC50 configurations
ec50_datasets = [
    'ec50_assay',
    'ec50_scaffold',
    'ec50_size',
]

def train_all_good():
    """
    Train on all GOOD dataset configurations SEQUENTIALLY (one by one).
    Trains 12 models total: GOODHIV (4), GOODZINC (4), GOODPCBA (4).
    Each dataset is trained separately with its own model.
    """
    print("="*80)
    print("Sequential Training: GOOD Datasets")
    print("="*80)
    print("Total: 12 configurations (GOODHIV: 4, GOODZINC: 4, GOODPCBA: 4)")
    print("Each dataset will be trained separately with its own model.")
    print("Estimated time: 4-8 hours (depending on hardware)")
    print("="*80 + "\n")

    cmd = [
        'python', 'train_dann.py',
        '--datasets'
    ] + good_datasets + [
        '--epochs', '100',
        '--batch_size', '32',
        '--patience', '20',
        '--results_file', 'dann_results_all_good.json'
    ]

    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def train_all_ic50():
    """
    Train on all DrugOOD IC50 configurations SEQUENTIALLY (one by one).
    Trains 3 models total: assay, scaffold, size.
    Each dataset is trained separately with its own model.
    """
    print("="*80)
    print("Sequential Training: DrugOOD IC50")
    print("="*80)
    print("Total: 3 configurations (assay, scaffold, size)")
    print("Each dataset will be trained separately with its own model.")
    print("Estimated time: 1-2 hours (depending on hardware)")
    print("="*80 + "\n")

    cmd = [
        'python', 'train_dann.py',
        '--datasets'
    ] + ic50_datasets + [
        '--epochs', '100',
        '--batch_size', '32',
        '--patience', '20',
        '--results_file', 'dann_results_all_ic50.json'
    ]

    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def train_all_ec50():
    """
    Train on all DrugOOD EC50 configurations SEQUENTIALLY (one by one).
    Trains 3 models total: assay, scaffold, size.
    Each dataset is trained separately with its own model.
    """
    print("="*80)
    print("Sequential Training: DrugOOD EC50")
    print("="*80)
    print("Total: 3 configurations (assay, scaffold, size)")
    print("Each dataset will be trained separately with its own model.")
    print("Estimated time: 1-2 hours (depending on hardware)")
    print("="*80 + "\n")

    cmd = [
        'python', 'train_dann.py',
        '--datasets'
    ] + ec50_datasets + [
        '--epochs', '100',
        '--batch_size', '32',
        '--patience', '20',
        '--results_file', 'dann_results_all_ec50.json'
    ]

    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def train_all():
    """
    Train on ALL dataset configurations SEQUENTIALLY (one by one).
    Trains 18 models total: GOOD (12) + IC50 (3) + EC50 (3).
    Each dataset is trained separately with its own model.
    WARNING: This will take a very long time!
    """
    print("="*80)
    print("Sequential Training: ALL Datasets")
    print("="*80)
    print("Total: 18 configurations (GOOD: 12, IC50: 3, EC50: 3)")
    print("Each dataset will be trained separately with its own model.")
    print("\nWARNING: This will take a VERY long time!")
    print("Estimated time: 6-12 hours (depending on hardware)")
    print("Consider training on subsets instead (good/ic50/ec50).\n")
    print("="*80 + "\n")

    all_datasets = good_datasets + ic50_datasets + ec50_datasets

    cmd = [
        'python', 'train_dann.py',
        '--datasets'
    ] + all_datasets + [
        '--epochs', '100',
        '--batch_size', '32',
        '--patience', '20',
        '--results_file', 'dann_results_all.json'
    ]

    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("="*80)
        print("DANN Sequential Training Script")
        print("="*80)
        print("\nUsage: python train_all_datasets.py [option]")
        print("\nOptions:")
        print("  good    - Train on all 12 GOOD configurations (4-8 hours)")
        print("  ic50    - Train on all 3 IC50 configurations (1-2 hours)")
        print("  ec50    - Train on all 3 EC50 configurations (1-2 hours)")
        print("  all     - Train on all 18 configurations (6-12 hours)")
        print("\nIMPORTANT:")
        print("  Each dataset is trained SEPARATELY, one after another.")
        print("  This script does NOT mix datasets in a single training run.")
        print("  Each dataset gets its own model and evaluation.")
        print("\nExamples:")
        print("  python train_all_datasets.py good    # Train all GOOD datasets")
        print("  python train_all_datasets.py ic50    # Train all IC50 datasets")
        print("="*80)
        sys.exit(1)

    option = sys.argv[1].lower()

    if option == 'good':
        train_all_good()
    elif option == 'ic50':
        train_all_ic50()
    elif option == 'ec50':
        train_all_ec50()
    elif option == 'all':
        train_all()
    else:
        print(f"Error: Unknown option '{option}'")
        print("Valid options: good, ic50, ec50, all")
        print("Run 'python train_all_datasets.py' for help.")
        sys.exit(1)
