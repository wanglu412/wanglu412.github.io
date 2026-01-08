"""
Quick test script to verify DANN implementation
"""

import torch
from dann_model import DANN, GradientReversalLayer, get_lambda_alpha


def test_gradient_reversal():
    """Test Gradient Reversal Layer"""
    print("Testing Gradient Reversal Layer...")

    grl = GradientReversalLayer()
    grl.set_lambda(1.0)

    # Create input with requires_grad
    x = torch.randn(10, 128, requires_grad=True)

    # Forward pass
    out = grl(x)

    # Check forward pass is identity
    assert torch.allclose(out, x), "Forward pass should be identity"

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check gradient is reversed
    expected_grad = -torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), "Gradient should be reversed"

    print("  ✓ Gradient Reversal Layer works correctly\n")


def test_lambda_schedule():
    """Test lambda scheduling"""
    print("Testing Lambda Schedule...")

    max_epochs = 100
    lambdas = [get_lambda_alpha(e, max_epochs) for e in range(max_epochs)]

    # Lambda should start near 0
    assert lambdas[0] < 0.1, "Lambda should start near 0"

    # Lambda should end near 1
    assert lambdas[-1] > 0.9, "Lambda should end near 1"

    # Lambda should be monotonically increasing
    for i in range(len(lambdas) - 1):
        assert lambdas[i] <= lambdas[i+1], "Lambda should be monotonically increasing"

    print(f"  Lambda at epoch 0: {lambdas[0]:.4f}")
    print(f"  Lambda at epoch 50: {lambdas[50]:.4f}")
    print(f"  Lambda at epoch 99: {lambdas[-1]:.4f}")
    print("  ✓ Lambda schedule works correctly\n")


def test_dann_model():
    """Test DANN model forward pass"""
    print("Testing DANN Model...")

    # Model parameters
    input_dim = 9  # Typical for molecular graphs
    num_classes = 1
    num_domains = 2
    hidden_dim = 64
    num_layers = 2

    # Create model
    model = DANN(
        input_dim=input_dim,
        num_classes=num_classes,
        num_domains=num_domains,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type='gin',
        dropout=0.5
    )

    model.eval()

    # Create dummy graph data
    num_nodes = 20
    num_edges = 40
    batch_size = 4

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[5:10] = 1
    batch[10:15] = 2
    batch[15:] = 3

    # Forward pass
    class_output, domain_output, features = model(x, edge_index, batch, alpha=0.5)

    # Check output shapes
    assert class_output.shape == (batch_size, num_classes), \
        f"Expected class output shape ({batch_size}, {num_classes}), got {class_output.shape}"

    assert domain_output.shape == (batch_size, num_domains), \
        f"Expected domain output shape ({batch_size}, {num_domains}), got {domain_output.shape}"

    assert features.shape == (batch_size, hidden_dim), \
        f"Expected features shape ({batch_size}, {hidden_dim}), got {features.shape}"

    print(f"  Input: {num_nodes} nodes, {num_edges} edges, {batch_size} graphs")
    print(f"  Class output shape: {class_output.shape}")
    print(f"  Domain output shape: {domain_output.shape}")
    print(f"  Features shape: {features.shape}")
    print("  ✓ DANN model works correctly\n")


def test_predict():
    """Test DANN prediction mode"""
    print("Testing DANN Prediction...")

    model = DANN(
        input_dim=9,
        num_classes=1,
        num_domains=2,
        hidden_dim=64,
        num_layers=2,
        gnn_type='gcn'
    )

    model.eval()

    # Create dummy data
    x = torch.randn(10, 9)
    edge_index = torch.randint(0, 10, (2, 20))
    batch = torch.zeros(10, dtype=torch.long)

    # Prediction without domain classification
    class_output = model.predict(x, edge_index, batch)

    assert class_output.shape == (1, 1), \
        f"Expected class output shape (1, 1), got {class_output.shape}"

    print(f"  Prediction output shape: {class_output.shape}")
    print("  ✓ DANN prediction works correctly\n")


def main():
    print("="*80)
    print("DANN Implementation Tests")
    print("="*80 + "\n")

    try:
        test_gradient_reversal()
        test_lambda_schedule()
        test_dann_model()
        test_predict()

        print("="*80)
        print("All tests passed! ✓")
        print("="*80 + "\n")

        print("The DANN implementation is working correctly.")
        print("You can now run: python train_dann.py")

    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}\n")
        raise
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}\n")
        raise


if __name__ == '__main__':
    main()
