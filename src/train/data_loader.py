from src.benchmark import L3SFV2AugmentedBenchmark, L3SFBenchmark, PolyUDBIIBenchmark
from src.gmdataset import RESCALE, GMDataset, get_dataloader


def build_dataloaders(
    train_root: str,
    dataset_len: int,
    batch_size: int,
    benchmark_name: str = "L3SFV2AugmentedBenchmark",
    filter=None,
    overfit_to_train_split: bool = False,
    stage: int = 4,
    has_dustbin: bool = True,
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
        only_genuine=True if stage == 1 else False,
    )

    test_bm = BM(
        sets=test_split,
        obj_resize=RESCALE,
        train_root=train_root,
        filter=filter,
        only_genuine=True if stage == 1 else False,
    )

    val_bm = BM(
        sets=val_split,
        obj_resize=RESCALE,
        train_root=train_root,
        filter=filter,
        only_genuine=True if stage == 1 else False,
    )

    ds_name = {
        "L3SFV2AugmentedBenchmark": "L3SFV2Augmented",
        "L3SFBenchmark": "L3SF",
        "PolyUDBIIBenchmark": "PolyUDBII"
    }[benchmark_name]
    

    image_dataset = GMDataset(ds_name, benchmark, dataset_len, True, None, "2GM", augment=True, has_dustbin=has_dustbin)
    test_dataset = GMDataset(ds_name, test_bm, dataset_len, True, None, "2GM", augment=False, has_dustbin=has_dustbin)
    val_dataset = GMDataset(ds_name, val_bm, dataset_len, True, None, "2GM", augment=False, has_dustbin=has_dustbin)

    dataloader = get_dataloader(image_dataset, batch_size=batch_size, shuffle=True, fix_seed=False, has_dustbin=has_dustbin)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, fix_seed=True, has_dustbin=has_dustbin)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, fix_seed=True, has_dustbin=has_dustbin)
    
    
    
    return dataloader, val_dataloader, test_dataloader
    # return val_dataloader, val_dataloader, val_dataloader

