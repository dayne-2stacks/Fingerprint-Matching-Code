import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
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
    univ_size: int = 300,
    rank: int = 0,
    world_size: int = 1,
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

    bm_kwargs = dict(obj_resize=RESCALE, train_root=train_root, filter=filter,
                     only_genuine=stage in (0, 1))

    # Rank 0 creates benchmark objects solely to trigger to_json() and write
    # the JSON index files to disk before other ranks read them.  The objects
    # are discarded immediately; all ranks rebuild their own copies after the
    # barrier so that per-process side-effects (e.g. gt_cache_path creation
    # inside BM.__init__ for the test split) never block the barrier.
    if rank == 0:
        _bm_train = BM(sets=train_split, **bm_kwargs)
        _bm_val   = BM(sets=val_split,   **bm_kwargs)
        _bm_test  = BM(sets=test_split,  **bm_kwargs)
        del _bm_train, _bm_val, _bm_test

    if world_size > 1:
        dist.barrier()

    # All ranks build their own benchmark objects.  JSON files now exist on
    # disk (rank 0 wrote them above) so to_json() returns immediately.
    benchmark = BM(sets=train_split, **bm_kwargs)
    val_bm    = BM(sets=val_split,   **bm_kwargs)
    test_bm   = BM(sets=test_split,  **bm_kwargs)

    ds_name = {
        "L3SFV2AugmentedBenchmark": "L3SFV2Augmented",
        "L3SFBenchmark": "L3SF",
        "PolyUDBIIBenchmark": "PolyUDBII"
    }[benchmark_name]
    

    image_dataset = GMDataset(ds_name, benchmark, dataset_len, True, None, "2GM", augment=True, has_dustbin=has_dustbin)
    test_dataset = GMDataset(ds_name, test_bm, dataset_len, True, None, "2GM", augment=False, has_dustbin=has_dustbin)
    val_dataset = GMDataset(ds_name, val_bm, dataset_len, True, None, "2GM", augment=False, has_dustbin=has_dustbin)

    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(image_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = get_dataloader(image_dataset, batch_size=batch_size, shuffle=(train_sampler is None), fix_seed=False, has_dustbin=has_dustbin, sampler=train_sampler)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, fix_seed=True, has_dustbin=has_dustbin)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, fix_seed=True, has_dustbin=has_dustbin)

    return dataloader, val_dataloader, test_dataloader, train_sampler
    # return val_dataloader, val_dataloader, val_dataloader

