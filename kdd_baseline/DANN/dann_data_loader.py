"""
Data loading and preprocessing utilities for DANN training
"""

import torch
from torch_geometric.loader import DataLoader
from datasets import DrugOODDataset, GOODHIV, GOODPCBA, GOODZINC
import numpy as np


def add_domain_labels(dataset, train_indices, val_indices, test_indices, domain_id):
    """
    Add domain labels to dataset

    Args:
        dataset: PyTorch Geometric dataset
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        domain_id: Domain ID to assign

    Returns:
        Modified dataset with domain labels
    """
    for idx in range(len(dataset)):
        if not hasattr(dataset[idx], 'domain'):
            dataset[idx].domain = torch.tensor([domain_id], dtype=torch.long)
    return dataset


def load_drugood_dataset(name, root="data", domain_id=0):
    """
    Load DrugOOD dataset (IC50 or EC50)

    Args:
        name: Dataset name (e.g., 'ic50_assay', 'ec50_scaffold')
        root: Root directory
        domain_id: Domain ID for this dataset

    Returns:
        train_dataset, val_dataset, test_dataset, num_node_features, num_classes
    """
    print(f"\nLoading DrugOOD dataset: {name}")
    dataset = DrugOODDataset(name=name, root=root)

    # Get splits
    train_indices = dataset.train_index
    val_indices = dataset.valid_index
    test_indices = dataset.test_index

    # Add domain labels
    dataset = add_domain_labels(dataset, train_indices, val_indices, test_indices, domain_id)

    # Create split datasets
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Get dataset info
    num_node_features = dataset[0].x.shape[1]
    num_classes = 1  # Binary classification

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Node features: {num_node_features}, Classes: {num_classes}")

    return train_dataset, val_dataset, test_dataset, num_node_features, num_classes


def load_good_dataset(dataset_class, dataset_name, root="data", domain_id=0, domain="scaffold", shift="covariate"):
    """
    Load GOOD dataset (GOODHIV, GOODPCBA, or GOODZINC)

    Args:
        dataset_class: Dataset class (GOODHIV, GOODPCBA, or GOODZINC)
        dataset_name: Dataset name for display
        root: Root directory
        domain_id: Domain ID for this dataset
        domain: Domain type ('scaffold' or 'size')
        shift: Shift type ('covariate', 'concept', or 'no_shift')

    Returns:
        train_dataset, val_dataset, test_dataset, num_node_features, num_classes
    """
    print(f"\nLoading {dataset_name} dataset (domain={domain}, shift={shift})")

    # Load with specified domain and shift
    splits, meta = dataset_class.load(
        dataset_root=root,
        domain=domain,
        shift=shift
    )

    # Extract splits
    train_dataset = list(splits['train'])
    val_dataset = list(splits['val'])
    test_dataset = list(splits['test'])

    # Add domain labels
    for data in train_dataset + val_dataset + test_dataset:
        data.domain = torch.tensor([domain_id], dtype=torch.long)

    # Get dataset info
    # GOOD datasets use 'dim_node' instead of 'num_node_features'
    if hasattr(meta, 'dim_node'):
        num_node_features = meta.dim_node
    elif hasattr(meta, 'num_node_features'):
        num_node_features = meta.num_node_features
    else:
        num_node_features = train_dataset[0].x.shape[1]

    if hasattr(meta, 'num_classes'):
        num_classes = meta.num_classes
    else:
        num_classes = 2

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Node features: {num_node_features}, Classes: {num_classes}")

    return train_dataset, val_dataset, test_dataset, num_node_features, num_classes


def create_multi_domain_loaders(datasets_config, batch_size=32, num_workers=0):
    """
    Create data loaders for multiple domains (for DANN training)

    Args:
        datasets_config: List of dataset configurations
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader, num_node_features, num_classes, num_domains
    """
    all_train = []
    all_val = []
    all_test = []
    num_node_features = None
    num_classes = None

    for config in datasets_config:
        dataset_type = config['type']
        domain_id = config['domain_id']

        if dataset_type == 'drugood':
            name = config['name']
            train_ds, val_ds, test_ds, nf, nc = load_drugood_dataset(
                name=name,
                root=config.get('root', 'data'),
                domain_id=domain_id
            )
        elif dataset_type == 'goodhiv':
            train_ds, val_ds, test_ds, nf, nc = load_good_dataset(
                GOODHIV,
                "GOODHIV",
                root=config.get('root', 'data'),
                domain_id=domain_id,
                domain=config.get('domain', 'scaffold'),
                shift=config.get('shift', 'covariate')
            )
        elif dataset_type == 'goodpcba':
            train_ds, val_ds, test_ds, nf, nc = load_good_dataset(
                GOODPCBA,
                "GOODPCBA",
                root=config.get('root', 'data'),
                domain_id=domain_id,
                domain=config.get('domain', 'scaffold'),
                shift=config.get('shift', 'covariate')
            )
        elif dataset_type == 'goodzinc':
            train_ds, val_ds, test_ds, nf, nc = load_good_dataset(
                GOODZINC,
                "GOODZINC",
                root=config.get('root', 'data'),
                domain_id=domain_id,
                domain=config.get('domain', 'scaffold'),
                shift=config.get('shift', 'covariate')
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        all_train.extend(train_ds)
        all_val.extend(val_ds)
        all_test.extend(test_ds)

        if num_node_features is None:
            num_node_features = nf
            num_classes = nc

    # Create data loaders
    train_loader = DataLoader(
        all_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        all_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        all_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    num_domains = len(datasets_config)

    print(f"\n{'='*80}")
    print(f"Multi-domain dataset created")
    print(f"{'='*80}")
    print(f"Total train samples: {len(all_train)}")
    print(f"Total val samples: {len(all_val)}")
    print(f"Total test samples: {len(all_test)}")
    print(f"Number of domains: {num_domains}")
    print(f"Node features: {num_node_features}")
    print(f"Number of classes: {num_classes}")
    print(f"{'='*80}\n")

    return train_loader, val_loader, test_loader, num_node_features, num_classes, num_domains


def create_single_dataset_loaders(dataset_config, batch_size=32, num_workers=0):
    """
    Create data loaders for a single dataset

    Args:
        dataset_config: Dataset configuration dict
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader, num_node_features, num_classes
    """
    dataset_type = dataset_config['type']

    if dataset_type == 'drugood':
        name = dataset_config['name']
        train_ds, val_ds, test_ds, num_node_features, num_classes = load_drugood_dataset(
            name=name,
            root=dataset_config.get('root', 'data'),
            domain_id=0
        )
    elif dataset_type == 'goodhiv':
        train_ds, val_ds, test_ds, num_node_features, num_classes = load_good_dataset(
            GOODHIV,
            "GOODHIV",
            root=dataset_config.get('root', 'data'),
            domain_id=0,
            domain=dataset_config.get('domain', 'scaffold'),
            shift=dataset_config.get('shift', 'covariate')
        )
    elif dataset_type == 'goodpcba':
        train_ds, val_ds, test_ds, num_node_features, num_classes = load_good_dataset(
            GOODPCBA,
            "GOODPCBA",
            root=dataset_config.get('root', 'data'),
            domain_id=0,
            domain=dataset_config.get('domain', 'scaffold'),
            shift=dataset_config.get('shift', 'covariate')
        )
    elif dataset_type == 'goodzinc':
        train_ds, val_ds, test_ds, num_node_features, num_classes = load_good_dataset(
            GOODZINC,
            "GOODZINC",
            root=dataset_config.get('root', 'data'),
            domain_id=0,
            domain=dataset_config.get('domain', 'scaffold'),
            shift=dataset_config.get('shift', 'covariate')
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, num_node_features, num_classes
