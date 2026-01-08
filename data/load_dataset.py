from datasets import DrugOODDataset, GOODHIV, GOODPCBA, GOODZINC
from torch_geometric.loader import DataLoader


def print_dataset_info(datasets_info):
    """
    Print dataset information in ASCII table format

    Parameters:
        datasets_info: List of dictionaries containing dataset information
    """
    # Define table headers
    headers = ["Dataset Name", "Total Graphs", "Train", "Val", "Test", "Node Feat Dim", "Num Classes", "Avg Nodes", "Avg Edges"]

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for info in datasets_info:
        col_widths[0] = max(col_widths[0], len(info['name']))
        col_widths[1] = max(col_widths[1], len(str(info['num_graphs'])))
        col_widths[2] = max(col_widths[2], len(str(info['train_size'])))
        col_widths[3] = max(col_widths[3], len(str(info['val_size'])))
        col_widths[4] = max(col_widths[4], len(str(info['test_size'])))
        col_widths[5] = max(col_widths[5], len(str(info['node_feat_dim'])))
        col_widths[6] = max(col_widths[6], len(str(info['num_classes'])))
        col_widths[7] = max(col_widths[7], len(str(info['avg_nodes'])))
        col_widths[8] = max(col_widths[8], len(str(info['avg_edges'])))

    # Print top border
    print("+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+")

    # Print headers
    header_row = "|"
    for i, header in enumerate(headers):
        header_row += f" {header:<{col_widths[i]}} |"
    print(header_row)

    # Print separator
    print("+" + "+".join(["=" * (w + 2) for w in col_widths]) + "+")

    # Print data rows
    for info in datasets_info:
        row = "|"
        row += f" {info['name']:<{col_widths[0]}} |"
        row += f" {info['num_graphs']:>{col_widths[1]}} |"
        row += f" {info['train_size']:>{col_widths[2]}} |"
        row += f" {info['val_size']:>{col_widths[3]}} |"
        row += f" {info['test_size']:>{col_widths[4]}} |"
        row += f" {info['node_feat_dim']:>{col_widths[5]}} |"
        row += f" {info['num_classes']:>{col_widths[6]}} |"
        row += f" {info['avg_nodes']:>{col_widths[7]}} |"
        row += f" {info['avg_edges']:>{col_widths[8]}} |"
        print(row)

    # Print bottom border
    print("+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+")


def get_dataset_stats(dataset, name, train_size=None, val_size=None, test_size=None):
    """
    Extract statistics from a dataset

    Parameters:
        dataset: PyTorch Geometric dataset or list of Data objects
        name: Name of the dataset
        train_size: Size of training set (optional)
        val_size: Size of validation set (optional)
        test_size: Size of test set (optional)

    Returns:
        Dictionary containing dataset statistics
    """
    # Convert to list if it's a dataset object
    if hasattr(dataset, '__getitem__') and not isinstance(dataset, list):
        data_list = [dataset[i] for i in range(len(dataset))]
    else:
        data_list = dataset

    num_graphs = len(data_list)

    # Get node feature dimension from first data point
    first_data = data_list[0]
    if hasattr(first_data, 'x') and first_data.x is not None:
        node_feat_dim = first_data.x.shape[1]
    else:
        node_feat_dim = 0

    # Get number of classes
    if hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    elif hasattr(first_data, 'y') and first_data.y is not None:
        y_shape = first_data.y.shape
        # Check if it's multi-label (vector with more than 1 element)
        if len(y_shape) > 0 and (y_shape[-1] > 1 or (len(y_shape) > 1 and y_shape[-1] > 1)):
            # Multi-label classification - use the last dimension
            num_classes = y_shape[-1] if len(y_shape) > 1 else y_shape[0]
        elif len(y_shape) == 0 or (len(y_shape) == 1 and y_shape[0] == 1):
            # Single-label classification - count unique labels
            all_labels = set()
            for data in data_list:
                if hasattr(data, 'y') and data.y is not None:
                    all_labels.add(int(data.y.item()))
            num_classes = len(all_labels)
        else:
            # Try to determine by checking if we can convert to scalar
            try:
                _ = first_data.y.item()
                # Single-label classification
                all_labels = set()
                for data in data_list:
                    if hasattr(data, 'y') and data.y is not None:
                        all_labels.add(int(data.y.item()))
                num_classes = len(all_labels)
            except:
                # Multi-label - use the size
                num_classes = first_data.y.numel() if len(y_shape) == 1 else y_shape[-1]
    else:
        num_classes = "N/A"

    # Calculate average number of nodes and edges
    total_nodes = 0
    total_edges = 0
    for data in data_list:
        total_nodes += data.num_nodes
        total_edges += data.edge_index.shape[1]

    avg_nodes = f"{total_nodes / num_graphs:.2f}"
    avg_edges = f"{total_edges / num_graphs:.2f}"

    return {
        'name': name,
        'num_graphs': num_graphs,
        'train_size': train_size if train_size is not None else "N/A",
        'val_size': val_size if val_size is not None else "N/A",
        'test_size': test_size if test_size is not None else "N/A",
        'node_feat_dim': node_feat_dim,
        'num_classes': num_classes,
        'avg_nodes': avg_nodes,
        'avg_edges': avg_edges
    }


if __name__ == "__main__":
    print("Loading datasets and collecting statistics...\n")

    datasets_info = []

    # Load all DrugOOD dataset variants
    drugood_variants = [
        "ic50_assay",
        "ic50_scaffold",
        "ic50_size",
        "ec50_assay",
        "ec50_scaffold",
        "ec50_size"
    ]

    for variant in drugood_variants:
        print(f"Loading DrugOOD ({variant})...")
        drugood_dataset = DrugOODDataset(name=variant, root="data")
        train_size = len(drugood_dataset.train_index)
        val_size = len(drugood_dataset.valid_index)
        test_size = len(drugood_dataset.test_index)
        datasets_info.append(get_dataset_stats(
            drugood_dataset,
            f"DrugOOD-{variant}",
            train_size=train_size,
            val_size=val_size,
            test_size=test_size
        ))

    # Load GOODHIV
    print("Loading GOODHIV...")
    goodhiv_splits, goodhiv_meta = GOODHIV.load(dataset_root="data", domain="scaffold", shift="covariate")
    # Combine all splits for statistics
    goodhiv_combined = []
    train_size = len(goodhiv_splits['train']) if goodhiv_splits['train'] is not None else 0
    val_size = len(goodhiv_splits['val']) if goodhiv_splits['val'] is not None else 0
    test_size = len(goodhiv_splits['test']) if goodhiv_splits['test'] is not None else 0

    for split_name in ['train', 'val', 'test']:
        if split_name in goodhiv_splits and goodhiv_splits[split_name] is not None:
            goodhiv_combined.extend(list(goodhiv_splits[split_name]))
    datasets_info.append(get_dataset_stats(
        goodhiv_combined,
        "GOODHIV",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    ))

    # Load GOODPCBA
    print("Loading GOODPCBA...")
    goodpcba_splits, goodpcba_meta = GOODPCBA.load(dataset_root="data", domain="scaffold", shift="covariate")
    # Combine all splits for statistics
    goodpcba_combined = []
    train_size = len(goodpcba_splits['train']) if goodpcba_splits['train'] is not None else 0
    val_size = len(goodpcba_splits['val']) if goodpcba_splits['val'] is not None else 0
    test_size = len(goodpcba_splits['test']) if goodpcba_splits['test'] is not None else 0

    for split_name in ['train', 'val', 'test']:
        if split_name in goodpcba_splits and goodpcba_splits[split_name] is not None:
            goodpcba_combined.extend(list(goodpcba_splits[split_name]))
    datasets_info.append(get_dataset_stats(
        goodpcba_combined,
        "GOODPCBA",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    ))

    # Load GOODZINC
    print("Loading GOODZINC...")
    goodzinc_splits, goodzinc_meta = GOODZINC.load(dataset_root="data", domain="scaffold", shift="covariate")
    # Combine all splits for statistics
    goodzinc_combined = []
    train_size = len(goodzinc_splits['train']) if goodzinc_splits['train'] is not None else 0
    val_size = len(goodzinc_splits['val']) if goodzinc_splits['val'] is not None else 0
    test_size = len(goodzinc_splits['test']) if goodzinc_splits['test'] is not None else 0

    for split_name in ['train', 'val', 'test']:
        if split_name in goodzinc_splits and goodzinc_splits[split_name] is not None:
            goodzinc_combined.extend(list(goodzinc_splits[split_name]))
    datasets_info.append(get_dataset_stats(
        goodzinc_combined,
        "GOODZINC",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    ))

    print("\n" + "="*120)
    print("DATASET STATISTICS")
    print("="*120 + "\n")

    # Print the table
    print_dataset_info(datasets_info)
