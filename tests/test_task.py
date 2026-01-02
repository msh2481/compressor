import torch
from task import all_boolean_inputs, xor_of_subset, make_dataset, get_all_xor_subsets


def test_all_boolean_inputs_shape():
    x = all_boolean_inputs(4)
    assert x.shape == (16, 4)


def test_all_boolean_inputs_values():
    x = all_boolean_inputs(3)
    # Should contain all 8 combinations of 3 bits
    assert x.shape == (8, 3)
    # Check specific row: index 5 = 101 in binary
    assert x[5].tolist() == [1, 0, 1]


def test_xor_of_subset_basic():
    x = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]]).float()
    y = xor_of_subset(x, k=2)
    expected = torch.tensor([0, 1, 1, 0, 0]).float()  # XOR of first 2 bits
    assert torch.equal(y, expected)


def test_xor_of_subset_all_bits():
    x = all_boolean_inputs(3)
    y = xor_of_subset(x, k=3)
    # XOR of all 3 bits: only odd number of 1s gives 1
    expected = x.sum(dim=1) % 2
    assert torch.equal(y, expected)


def test_make_dataset_no_overlap():
    train_x, _, val_x, _ = make_dataset(n=6, k=3, train_fraction=0.8, seed=42)

    # Convert to sets of tuples for comparison
    train_set = set(tuple(row.tolist()) for row in train_x)
    val_set = set(tuple(row.tolist()) for row in val_x)

    assert len(train_set & val_set) == 0, "Train and val sets overlap!"


def test_make_dataset_covers_all():
    train_x, _, val_x, _ = make_dataset(n=6, k=3, train_fraction=0.8, seed=42)

    # Total should be 2^6 = 64
    assert len(train_x) + len(val_x) == 64


def test_make_dataset_split_ratio():
    train_x, _, val_x, _ = make_dataset(n=6, k=3, train_fraction=0.8, seed=42)

    assert len(train_x) == 51  # 0.8 * 64 = 51.2 -> 51
    assert len(val_x) == 13


def test_make_dataset_labels_correct():
    train_x, train_y, val_x, val_y = make_dataset(n=6, k=3, train_fraction=0.8, seed=42)

    # Verify labels match XOR computation
    expected_train_y = xor_of_subset(train_x, k=3)
    expected_val_y = xor_of_subset(val_x, k=3)

    assert torch.equal(train_y, expected_train_y)
    assert torch.equal(val_y, expected_val_y)


def test_make_dataset_deterministic():
    data1 = make_dataset(n=6, k=3, seed=123)
    data2 = make_dataset(n=6, k=3, seed=123)

    for t1, t2 in zip(data1, data2):
        assert torch.equal(t1, t2)


def test_get_all_xor_subsets_count():
    subsets = get_all_xor_subsets(4)
    # 2^4 - 1 = 15 non-empty subsets
    assert len(subsets) == 15


def test_get_all_xor_subsets_correctness():
    subsets = get_all_xor_subsets(3)
    all_x = all_boolean_inputs(3)

    # Check subset (0, 1) = XOR of bits 0 and 1
    expected = (all_x[:, 0] + all_x[:, 1]) % 2
    assert torch.equal(subsets[(0, 1)], expected)

    # Check subset (0, 1, 2) = XOR of all bits
    expected_all = (all_x[:, 0] + all_x[:, 1] + all_x[:, 2]) % 2
    assert torch.equal(subsets[(0, 1, 2)], expected_all)
