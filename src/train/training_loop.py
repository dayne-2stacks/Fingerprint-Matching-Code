import torch
import torch.nn.functional as F
from itertools import islice
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import save_model
from src.train.common import (
    batch_match_counts,
    compose_total_loss,
    counts_summary,
    matching_metrics_from_counts,
    summary_scalars,
)


def _save_mid_epoch(model, optimizer, optimizer_k, epoch, batch, checkpoint_path):
    from utils.models_sl import save_model as _sm
    from torch.nn.parallel import DistributedDataParallel
    from torch.nn import DataParallel
    m = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    torch.save({
        "epoch": epoch,
        "batch": batch,
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_k": optimizer_k.state_dict() if optimizer_k is not None else None,
    }, str(checkpoint_path / "mid_epoch.pt"))


def train_epoch(model, dataloader, criterion, optimizer, optimizer_k,
                device, writer, epoch, start_epoch, stage, logger, checkpoint_path,
                detect_anomaly=False, max_iters=None, is_main=True,
                start_batch=0, save_every=50):
    # Initialize running sums and counters
    epoch_loss_sum = 0.0
    running_ks_loss = 0.0
    running_ks_error = 0.0
    running_dustbin_loss = 0.0
    epoch_total_loss_sum = 0.0
    metric_tp_sum = 0.0
    metric_tn_sum = 0.0
    metric_fp_sum = 0.0
    metric_fn_sum = 0.0
    iter_num = 0

    torch.autograd.set_detect_anomaly(bool(detect_anomaly))

    it = iter(islice(dataloader, max_iters))
    # Skip already-completed batches
    if start_batch > 0:
        print(f"Resuming epoch {epoch} from batch {start_batch}, skipping ahead...")
        for _ in range(start_batch):
            try:
                next(it)
            except StopIteration:
                break

    # One dataloader pass per epoch
    for batch_idx, batch in enumerate(it, start=start_batch):
        iter_num += 1

        batch = data_to_cuda(batch)
        
        optimizer.zero_grad()
        if optimizer_k is not None:
            optimizer_k.zero_grad()
        
        if stage in (0, 1):
            outputs = model(batch, regression=False)
        else:
            outputs = model(batch, regression=True)

        loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])

        ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
        ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))

        dustbin_loss = torch.tensor(0.0, device=device)
        ss_db = outputs.get("ds_mat_db")
        if ss_db is not None:
            gt_perm = outputs["gt_perm_mat"]
            n1s, n2s = outputs["ns"]
            eps = 1e-15
            n_valid = 0
            for b in range(ss_db.shape[0]):
                n1_b = int(n1s[b])
                n2_b = int(n2s[b])
                if n1_b == 0 or n2_b == 0:
                    continue
                gt_db_col = (1.0 - gt_perm[b, :n1_b, :n2_b].sum(dim=1)).clamp(0.0, 1.0)
                gt_db_row = (1.0 - gt_perm[b, :n1_b, :n2_b].sum(dim=0)).clamp(0.0, 1.0)
                pred_db_col = ss_db[b, :n1_b, n2_b].clamp(eps, 1 - eps)
                pred_db_row = ss_db[b, n1_b, :n2_b].clamp(eps, 1 - eps)
                dustbin_loss = dustbin_loss + F.binary_cross_entropy(pred_db_col, gt_db_col, reduction='mean')
                dustbin_loss = dustbin_loss + F.binary_cross_entropy(pred_db_row, gt_db_row, reduction='mean')
                n_valid += 1
            if n_valid > 0:
                dustbin_loss = dustbin_loss / n_valid

        total_loss = compose_total_loss(
            primary_loss=loss,
            ks_loss=ks_loss,
            dustbin_loss=dustbin_loss,
            stage=stage,
        )

        if not torch.isfinite(total_loss).all():
            print(f"[WARN] Non-finite loss at iter {iter_num}, skipping batch.")
            logger.warning("Non-finite loss at iter %s, skipping batch.", iter_num)
            continue

        loss_value = loss.item()
        ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
        total_loss_value = total_loss.item()
        total_loss.backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        optimizer.step()
        if optimizer_k is not None:
            optimizer_k.step()

        epoch_loss_sum += loss_value
        epoch_total_loss_sum += total_loss_value
        running_ks_loss += ks_loss_value
        running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error
        running_dustbin_loss += dustbin_loss.item() if isinstance(dustbin_loss, torch.Tensor) else dustbin_loss

        batch_counts = batch_match_counts(outputs)
        batch_tp = batch_counts["tp"]
        batch_tn = batch_counts["tn"]
        batch_fp = batch_counts["fp"]
        batch_fn = batch_counts["fn"]

        print(f"Batch {batch_idx}: \n TP={batch_tp}, \n TN={batch_tn}, \n FP={batch_fp}, \n FN={batch_fn}")
        metric_tp_sum += batch_tp
        metric_tn_sum += batch_tn
        metric_fp_sum += batch_fp
        metric_fn_sum += batch_fn

        batch_summary = counts_summary(batch_tp, batch_tn, batch_fp, batch_fn, device)
        batch_scalars = summary_scalars(batch_summary)
        acc = batch_scalars["accuracy"]
        prec = batch_scalars["precision"]
        rec = batch_scalars["recall"]
        f1 = batch_scalars["f1"]
        micro_f1 = batch_scalars["micro_f1"]
        macro_f1 = batch_scalars["macro_f1"]

        if is_main and iter_num % save_every == 0:
            _save_mid_epoch(model, optimizer, optimizer_k, epoch, batch_idx + 1, checkpoint_path)

        if iter_num % 5 == 0:
            avg_loss = epoch_loss_sum / iter_num
            avg_ks_loss = running_ks_loss / iter_num
            avg_dustbin_loss = running_dustbin_loss / iter_num
            avg_total_loss = epoch_total_loss_sum / iter_num

            log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                    f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                    f"db_loss: {avg_dustbin_loss:.4f}, "
                    f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}, "
                    f"P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}, "
                    f"MiF1: {micro_f1:.4f}, MaF1: {macro_f1:.4f}")
            print(log_msg)
            logger.info(log_msg)

    denom = max(iter_num, 1)
    avg_epoch_loss = epoch_loss_sum / denom
    avg_ks_loss = running_ks_loss / denom
    avg_dustbin_loss = running_dustbin_loss / denom
    avg_total_loss = epoch_total_loss_sum / denom
    epoch_metric_summary = matching_metrics_from_counts(
        torch.tensor(metric_tp_sum, device=device),
        torch.tensor(metric_tn_sum, device=device),
        torch.tensor(metric_fp_sum, device=device),
        torch.tensor(metric_fn_sum, device=device),
    )
    avg_accuracy = float(epoch_metric_summary["accuracy"].item())
    avg_precision = float(epoch_metric_summary["precision"].item())
    avg_recall = float(epoch_metric_summary["recall"].item())
    avg_f1 = float(epoch_metric_summary["f1"].item())
    avg_micro_f1 = float(epoch_metric_summary["micro_f1"].item())
    avg_macro_f1 = float(epoch_metric_summary["macro_f1"].item())


    log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
            f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Dustbin Loss: {avg_dustbin_loss:.4f}, "
            f"Avg Total Loss: {avg_total_loss:.4f}, Acc: {avg_accuracy:.4f}, "
            f"P: {avg_precision:.4f}, R: {avg_recall:.4f}, F1: {avg_f1:.4f}, "
            f"MiF1: {avg_micro_f1:.4f}, MaF1: {avg_macro_f1:.4f}")
    print(log_msg)
    logger.info(log_msg)

    # Remove mid-epoch checkpoint — epoch completed cleanly
    if is_main:
        mid = checkpoint_path / "mid_epoch.pt"
        if mid.exists():
            mid.unlink()

    # Save model and optimizer states (rank 0 only)
    if is_main:
        save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
        torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
        if optimizer_k is not None:
            torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))

        # Delete the previous epoch's numbered checkpoints — best_model.pt is kept separately
        for stem in (f"params_{epoch:04}.pt", f"optim_{epoch:04}.pt", f"optim_k_{epoch:04}.pt"):
            old = checkpoint_path / stem
            if old.exists():
                old.unlink()

    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
