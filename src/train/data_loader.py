from src.benchmark import L3SFV2AugmentedBenchmark, L3SFBenchmark, PolyUDBIIBenchmark
from src.gmdataset import RESCALE, GMDataset, get_dataloader


def build_dataloaders(
    train_root: str,
    dataset_len: int,
    batch_size: int,
    benchmark_name: str = "L3SFV2AugmentedBenchmark",
    filter=None,
    overfit_to_train_split: bool = False,
):
    """Create dataloaders for training, validation and testing.

    """
    BM = {
        "L3SFV2AugmentedBenchmark": L3SFV2AugmentedBenchmark,
        "L3SFBenchmark": L3SFBenchmark,
        "PolyUDBIIBenchmark": PolyUDBIIBenchmark
    }[benchmark_name]

    train_split = 'train'
    val_split = 'train' if overfit_to_train_split else 'val'
    test_split = 'train' if overfit_to_train_split else 'test'

    benchmark = BM(
        sets=train_split,
        obj_resize=RESCALE,
        train_root=train_root,
        filter=filter,
    )

    test_bm = BM(
        sets=test_split,
        obj_resize=RESCALE,
        train_root=train_root,
        filter=filter,
    )

    val_bm = BM(
        sets=val_split,
        obj_resize=RESCALE,
        train_root=train_root,
        filter=filter,
    )

    ds_name = {
        "L3SFV2AugmentedBenchmark": "L3SFV2Augmented",
        "L3SFBenchmark": "L3SF",
        "PolyUDBIIBenchmark": "PolyUDBII"
    }[benchmark_name]
    
    # In explicit overfit mode, disable train augmentation and reuse the train split
    # for val/test to make memorization behavior observable.
    train_augment = not overfit_to_train_split
    image_dataset = GMDataset(ds_name, benchmark, dataset_len, True, None, "2GM", augment=train_augment)
    test_dataset = GMDataset(ds_name, test_bm, dataset_len, True, None, "2GM", augment=False)
    val_dataset = GMDataset(ds_name, val_bm, dataset_len, True, None, "2GM", augment=False)

    dataloader = get_dataloader(image_dataset, batch_size=batch_size, shuffle=True, fix_seed=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, fix_seed=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, fix_seed=True)

    return dataloader, val_dataloader, test_dataloader
