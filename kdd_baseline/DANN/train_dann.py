"""
Main training script for DANN on multiple datasets
Trains DANN on GOODHIV, GOODZINC, EC50, and IC50 datasets
"""

import sys
import os
# Add parent directory to path to import datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import json
from datetime import datetime
import logging

from dann_model import DANN
from dann_trainer import DANNTrainer
from dann_data_loader import create_single_dataset_loaders


def setup_logger(log_dir, dataset_name):
    """
    Setup logger to write to both console and file

    Args:
        log_dir: Directory to save log files
        dataset_name: Name of the dataset being trained

    Returns:
        logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename WITHOUT timestamp (one file per dataset)
    log_file = os.path.join(log_dir, f'dann_{dataset_name}.log')

    # Configure logger
    logger = logging.getLogger('DANN')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler with append mode ('a') to keep all runs in one file
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Add a separator for new run
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"\n{'='*80}")
    logger.info(f"NEW TRAINING RUN - {timestamp}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}\n")

    return logger


def train_on_dataset(dataset_config, args, device, logger):
    """
    Train DANN on a single dataset

    Args:
        dataset_config: Dataset configuration
        args: Command line arguments
        device: Device to train on
        logger: Logger for recording training progress

    Returns:
        Results dictionary
    """
    dataset_name = dataset_config.get('display_name', dataset_config['type'])
    logger.info(f"\n{'='*80}")
    logger.info(f"Training DANN on {dataset_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, num_node_features, num_classes = \
        create_single_dataset_loaders(
            dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Create model
    logger.info("Creating DANN model...")
    model = DANN(
        input_dim=num_node_features,
        num_classes=num_classes,
        num_domains=args.num_domains,  # Set to 1 for single domain
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        dropout=args.dropout
    )
    logger.info(f"Model parameters: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, gnn_type={args.gnn_type}")

    # Create trainer
    trainer = DANNTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logger=logger
    )

    # Train
    save_path = os.path.join(args.save_dir, f"dann_{dataset_name.lower().replace(' ', '_')}.pt")
    logger.info(f"Starting training, model will be saved to: {save_path}")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        domain_weight=args.domain_weight,
        early_stopping_patience=args.patience,
        save_path=save_path
    )

    results['dataset'] = dataset_name
    return results


def main():
    parser = argparse.ArgumentParser(description='Train DANN on multiple datasets')

    # Dataset arguments
    parser.add_argument('--dataset', type=str,
                        default='goodhiv_scaffold_covariate',
                        help='Dataset to train on (single dataset per training run)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=None,
                        help='(Legacy) Use --dataset instead. This will train multiple datasets sequentially.')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory for datasets (relative to DANN folder)')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gin',
                        choices=['gin', 'gcn'],
                        help='Type of GNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--num_domains', type=int, default=1,
                        help='Number of domains (for single dataset training)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--domain_weight', type=float, default=1.0,
                        help='Weight for domain classification loss')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='dann_checkpoints',
                        help='Directory to save models')
    parser.add_argument('--results_file', type=str, default='dann_results.json',
                        help='File to save results')
    parser.add_argument('--log_dir', type=str, default='../log',
                        help='Directory to save log files (relative to DANN folder)')

    args = parser.parse_args()

    # Handle legacy --datasets argument
    if args.datasets is not None:
        print("Warning: --datasets is deprecated. Use --dataset for single dataset training,")
        print("         or use train_all_datasets.py to train on multiple datasets sequentially.")
        print(f"Training on datasets sequentially: {args.datasets}\n")
        datasets_to_train = args.datasets
    else:
        # Single dataset mode (recommended)
        datasets_to_train = [args.dataset]

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nUsing device: {device}")

    # Dataset configurations
    dataset_configs = {
        # GOODHIV configurations (scaffold + size) x (covariate + concept)
        'goodhiv_scaffold_covariate': {
            'type': 'goodhiv',
            'display_name': 'GOODHIV_Scaffold_Covariate',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },
        'goodhiv_scaffold_concept': {
            'type': 'goodhiv',
            'display_name': 'GOODHIV_Scaffold_Concept',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'concept'
        },
        'goodhiv_size_covariate': {
            'type': 'goodhiv',
            'display_name': 'GOODHIV_Size_Covariate',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'covariate'
        },
        'goodhiv_size_concept': {
            'type': 'goodhiv',
            'display_name': 'GOODHIV_Size_Concept',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'concept'
        },

        # GOODZINC configurations (scaffold + size) x (covariate + concept)
        'goodzinc_scaffold_covariate': {
            'type': 'goodzinc',
            'display_name': 'GOODZINC_Scaffold_Covariate',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },
        'goodzinc_scaffold_concept': {
            'type': 'goodzinc',
            'display_name': 'GOODZINC_Scaffold_Concept',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'concept'
        },
        'goodzinc_size_covariate': {
            'type': 'goodzinc',
            'display_name': 'GOODZINC_Size_Covariate',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'covariate'
        },
        'goodzinc_size_concept': {
            'type': 'goodzinc',
            'display_name': 'GOODZINC_Size_Concept',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'concept'
        },

        # GOODPCBA configurations (scaffold + size) x (covariate + concept)
        'goodpcba_scaffold_covariate': {
            'type': 'goodpcba',
            'display_name': 'GOODPCBA_Scaffold_Covariate',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },
        'goodpcba_scaffold_concept': {
            'type': 'goodpcba',
            'display_name': 'GOODPCBA_Scaffold_Concept',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'concept'
        },
        'goodpcba_size_covariate': {
            'type': 'goodpcba',
            'display_name': 'GOODPCBA_Size_Covariate',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'covariate'
        },
        'goodpcba_size_concept': {
            'type': 'goodpcba',
            'display_name': 'GOODPCBA_Size_Concept',
            'root': args.data_root,
            'domain': 'size',
            'shift': 'concept'
        },

        # Legacy short names for backward compatibility
        'goodhiv': {
            'type': 'goodhiv',
            'display_name': 'GOODHIV',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },
        'goodzinc': {
            'type': 'goodzinc',
            'display_name': 'GOODZINC',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },
        'goodpcba': {
            'type': 'goodpcba',
            'display_name': 'GOODPCBA',
            'root': args.data_root,
            'domain': 'scaffold',
            'shift': 'covariate'
        },

        # DrugOOD IC50 configurations
        'ic50_assay': {
            'type': 'drugood',
            'name': 'ic50_assay',
            'display_name': 'IC50_Assay',
            'root': args.data_root
        },
        'ic50_scaffold': {
            'type': 'drugood',
            'name': 'ic50_scaffold',
            'display_name': 'IC50_Scaffold',
            'root': args.data_root
        },
        'ic50_size': {
            'type': 'drugood',
            'name': 'ic50_size',
            'display_name': 'IC50_Size',
            'root': args.data_root
        },

        # DrugOOD EC50 configurations
        'ec50_assay': {
            'type': 'drugood',
            'name': 'ec50_assay',
            'display_name': 'EC50_Assay',
            'root': args.data_root
        },
        'ec50_scaffold': {
            'type': 'drugood',
            'name': 'ec50_scaffold',
            'display_name': 'EC50_Scaffold',
            'root': args.data_root
        },
        'ec50_size': {
            'type': 'drugood',
            'name': 'ec50_size',
            'display_name': 'EC50_Size',
            'root': args.data_root
        }
    }

    # Train on each dataset (sequentially, one at a time)
    all_results = []

    if len(datasets_to_train) > 1:
        print(f"\n{'='*80}")
        print(f"Sequential Training Mode: Will train {len(datasets_to_train)} datasets one by one")
        print(f"{'='*80}\n")

    for dataset_key in datasets_to_train:
        if dataset_key not in dataset_configs:
            print(f"Warning: Unknown dataset '{dataset_key}', skipping...")
            continue

        try:
            # Create logger for this dataset
            dataset_display_name = dataset_configs[dataset_key].get('display_name', dataset_key)
            logger = setup_logger(args.log_dir, dataset_display_name)

            results = train_on_dataset(
                dataset_configs[dataset_key],
                args,
                device,
                logger
            )
            all_results.append(results)

            # Print summary for this dataset
            print(f"\n{'='*80}")
            print(f"Results for {results['dataset']}")
            print(f"{'='*80}")
            print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
            print(f"Test AUC: {results['test_auc']:.4f}")
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Training Time: {results['training_time']/60:.2f} minutes")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\nError training on {dataset_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save all results
    if all_results:
        results_data = {
            'args': vars(args),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results
        }

        with open(args.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Print final summary
        if len(all_results) == 1:
            # Single dataset training
            result = all_results[0]
            print(f"\n{'='*80}")
            print(f"Training Complete - {result['dataset']}")
            print(f"{'='*80}")
            print(f"Best Validation AUC: {result['best_val_auc']:.4f}")
            print(f"Test AUC: {result['test_auc']:.4f}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"Training Time: {result['training_time']/60:.2f} minutes")
            print(f"{'='*80}")
            print(f"\nResults saved to: {args.results_file}")
            print(f"Model saved to: {args.save_dir}/")
            print(f"{'='*80}\n")
        else:
            # Multiple datasets trained sequentially
            print(f"\n{'='*80}")
            print("FINAL SUMMARY - Sequential Training on Multiple Datasets")
            print(f"{'='*80}")
            print(f"{'Dataset':<30} {'Val AUC':<12} {'Test AUC':<12} {'Test Acc':<12}")
            print(f"{'-'*80}")

            for result in all_results:
                print(f"{result['dataset']:<30} "
                      f"{result['best_val_auc']:<12.4f} "
                      f"{result['test_auc']:<12.4f} "
                      f"{result['test_accuracy']:<12.4f}")

            print(f"{'='*80}")
            print(f"\nResults saved to: {args.results_file}")
            print(f"Models saved to: {args.save_dir}/")
            print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
