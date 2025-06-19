from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader


def build_dataloaders(train_root: str, dataset_len: int):
    """Create dataloaders for training, validation and testing."""
    benchmark = L3SFV2AugmentedBenchmark(
        sets='train',
        obj_resize=(320, 240),
        train_root=train_root
    )

    test_bm = L3SFV2AugmentedBenchmark(
        sets='test',
        obj_resize=(320, 240),
        train_root=train_root
    )

    val_bm = L3SFV2AugmentedBenchmark(
        sets='val',
        obj_resize=(320, 240),
        train_root=train_root
    )

    image_dataset = GMDataset("L3SFV2Augmented", benchmark, dataset_len, True, None, "2GM")
    test_dataset = GMDataset("L3SFV2Augmented", test_bm, dataset_len, True, None, "2GM")
    val_dataset = GMDataset("L3SFV2Augmented", val_bm, dataset_len, True, None, "2GM")

    dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=False)
    test_dataloader = get_dataloader(test_dataset, shuffle=True, fix_seed=False)
    val_dataloader = get_dataloader(val_dataset, shuffle=True, fix_seed=False)

    return dataloader, val_dataloader, test_dataloader
