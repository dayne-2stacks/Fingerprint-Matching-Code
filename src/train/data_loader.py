from src.benchmark import L3SFV2AugmentedBenchmark, L3SFBenchmark
from src.gmdataset import GMDataset, get_dataloader


def build_dataloaders(train_root: str, dataset_len: int, task: str = 'match', dataset_kind: str = 'aug'):
    """Create dataloaders for training, validation and testing.

    - dataset_kind='aug' uses L3SFV2AugmentedBenchmark (Synthetic)
    - dataset_kind='l3sf' uses L3SFBenchmark (session/identity pairing)
    """
    BM = L3SFV2AugmentedBenchmark if dataset_kind != 'l3sf' else L3SFBenchmark

    benchmark = BM(
        sets='train',
        obj_resize=(320, 240),
        train_root=train_root,
        task=task
    )

    test_bm = BM(
        sets='test',
        obj_resize=(320, 240),
        train_root=train_root,
        task=task
    )

    val_bm = BM(
        sets='val',
        obj_resize=(320, 240),
        train_root=train_root,
        task=task
    )

    ds_name = "L3SF" if dataset_kind == 'l3sf' else "L3SFV2Augmented"
    image_dataset = GMDataset(ds_name, benchmark, dataset_len, True, None, "2GM", augment=True)
    test_dataset = GMDataset(ds_name, test_bm, dataset_len, True, None, "2GM", augment=False)
    val_dataset = GMDataset(ds_name, val_bm, dataset_len, True, None, "2GM", augment=False)

    dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=False)
    test_dataloader = get_dataloader(test_dataset, shuffle=True, fix_seed=False)
    val_dataloader = get_dataloader(val_dataset, shuffle=True, fix_seed=False)

    return dataloader, val_dataloader, test_dataloader
