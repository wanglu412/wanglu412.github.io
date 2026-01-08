"""
List all available dataset configurations for DANN training
"""

print("="*80)
print("AVAILABLE DATASET CONFIGURATIONS FOR DANN TRAINING")
print("="*80)

print("\n1. GOOD DATASETS (12 configurations)")
print("-" * 80)

print("\n   GOODHIV - HIV Activity Prediction (4 configurations):")
print("   - goodhiv_scaffold_covariate")
print("   - goodhiv_scaffold_concept")
print("   - goodhiv_size_covariate")
print("   - goodhiv_size_concept")

print("\n   GOODZINC - Molecular Property Prediction (4 configurations):")
print("   - goodzinc_scaffold_covariate")
print("   - goodzinc_scaffold_concept")
print("   - goodzinc_size_covariate")
print("   - goodzinc_size_concept")

print("\n   GOODPCBA - Bioactivity Prediction (4 configurations):")
print("   - goodpcba_scaffold_covariate")
print("   - goodpcba_scaffold_concept")
print("   - goodpcba_size_covariate")
print("   - goodpcba_size_concept")

print("\n2. DRUGOOD IC50 DATASETS (3 configurations)")
print("-" * 80)
print("   - ic50_assay    (Assay-based split)")
print("   - ic50_scaffold (Scaffold-based split)")
print("   - ic50_size     (Size-based split)")

print("\n3. DRUGOOD EC50 DATASETS (3 configurations)")
print("-" * 80)
print("   - ec50_assay    (Assay-based split)")
print("   - ec50_scaffold (Scaffold-based split)")
print("   - ec50_size     (Size-based split)")

print("\n4. LEGACY SHORT NAMES (for backward compatibility)")
print("-" * 80)
print("   - goodhiv   (alias for goodhiv_scaffold_covariate)")
print("   - goodzinc  (alias for goodzinc_scaffold_covariate)")
print("   - goodpcba  (alias for goodpcba_scaffold_covariate)")

print("\n" + "="*80)
print("TOTAL: 18 unique dataset configurations")
print("="*80)

print("\nEXAMPLE USAGE:")
print("-" * 80)
print("\n# Train on default datasets:")
print("python train_dann.py")

print("\n# Train on all GOODHIV configurations:")
print("python train_dann.py --datasets goodhiv_scaffold_covariate goodhiv_scaffold_concept goodhiv_size_covariate goodhiv_size_concept")

print("\n# Train on all GOOD datasets (12 configurations):")
print("python train_all_datasets.py good")

print("\n# Train on all IC50 datasets:")
print("python train_dann.py --datasets ic50_assay ic50_scaffold ic50_size")

print("\n# Train on specific mix:")
print("python train_dann.py --datasets goodhiv_scaffold_covariate goodzinc_size_concept ic50_assay")

print("\n" + "="*80 + "\n")
