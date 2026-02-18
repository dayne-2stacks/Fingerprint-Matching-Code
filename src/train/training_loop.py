import torch
import torch.nn.functional as F
from itertools import islice
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import save_model
from src.evaluation_metric import matching_accuracy
from src.model.dustbin import strip_dustbin_by_ns



def _should_strip_dustbin(outputs):
    if outputs.get("has_dustbin", False):
        return True
    gt = outputs.get("gt_perm_mat")
    if gt is None:
        return False
    if gt.ndim < 3 or gt.shape[-2] < 2 or gt.shape[-1] < 2:
        return False
    row_sum = gt[..., -1, :].sum(dim=-1)
    col_sum = gt[..., :, -1].sum(dim=-2)
    return bool((row_sum > 1).any() or (col_sum > 1).any())


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _grad_group_params(model):
    core_model = _unwrap_model(model)
    groups = {
        "Matcher": [],
        "KHead": [],

        "Dustbin": [],
    }
    for name, param in core_model.named_parameters():
        if name.startswith(("encoder_k.", "final_row.", "final_col.")):
            groups["KHead"].append(param)
        elif name == "bin_score":
            groups["Dustbin"].append(param)
        else:
            groups["Matcher"].append(param)
    return groups


def _group_grad_stats(model):
    group_params = _grad_group_params(model)
    norms = {}
    has_grad = {}
    has_nonfinite = {}

    for group_name, params in group_params.items():
        sq_norm = 0.0
        found_grad = False
        found_nonfinite = False
        for param in params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            found_grad = True
            grad_det = grad.detach()
            if not torch.isfinite(grad_det).all():
                found_nonfinite = True
                grad_det = torch.nan_to_num(grad_det, nan=0.0, posinf=0.0, neginf=0.0)
            sq_norm += float(torch.sum(grad_det * grad_det).item())

        norms[group_name] = sq_norm ** 0.5
        has_grad[group_name] = found_grad
        has_nonfinite[group_name] = found_nonfinite

    return norms, has_grad, has_nonfinite


def _loss_as_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), device=device)


def _compose_total_loss(
    primary_loss,
    ks_loss,
    dustbin_loss,
    dustbin_margin_loss,
    dustbin_margin_loss_weight,
):
    device = primary_loss.device
    return (
        _loss_as_tensor(primary_loss, device)
        + _loss_as_tensor(ks_loss, device)
        + _loss_as_tensor(dustbin_loss, device)
        + _loss_as_tensor(dustbin_margin_loss, device) * float(dustbin_margin_loss_weight)
    )



def train_epoch(model, dataloader, criterion, optimizer, optimizer_k,
                device, writer, epoch, start_epoch, stage, logger, checkpoint_path,
                detect_anomaly=False, max_iters=None):
    # Initialize running sums and counters
    epoch_loss_sum = 0.0
    running_ks_loss = 0.0
    running_dustbin_loss = 0.0
    running_dustbin_margin_loss = 0.0
    running_ks_error = 0.0
    epoch_accuracy_sum = 0.0
    epoch_total_loss_sum = 0.0
    iter_num = 0
    grad_norm_accum = {"Matcher": 0.0, "KHead": 0.0, "Dustbin": 0.0}
    grad_norm_count = {"Matcher": 0, "KHead": 0,  "Dustbin": 0}
    zero_grad_streak = {"Matcher": 0, "KHead": 0,  "Dustbin": 0}

    torch.autograd.set_detect_anomaly(bool(detect_anomaly))
    for i in range(3):
        steps_per_epoch = max_iters if max_iters is not None else len(dataloader)
        for batch_idx, batch in enumerate(islice(dataloader, max_iters)):
            iter_num += 1

            # Send data to device
            batch = data_to_cuda(batch)

            # Zero gradients
            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()
                
            # Forward pass
            outputs = model(batch)

            if _should_strip_dustbin(outputs):
                n1 = outputs["ns"][0]
                n2 = outputs["ns"][1]
                if outputs.get("has_dustbin", False):
                    n1 = n1 - 1
                    n2 = n2 - 1
                outputs["ds_mat"] = strip_dustbin_by_ns(outputs["ds_mat"], n1, n2)
                outputs["perm_mat"] = strip_dustbin_by_ns(outputs["perm_mat"], n1, n2)
                if "gt_perm_mat" in outputs:
                    outputs["gt_perm_mat"] = strip_dustbin_by_ns(outputs["gt_perm_mat"], n1, n2)
                outputs["ns"] = [n1, n2]


            # compute loss and their gradients
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))
            dustbin_loss = outputs.get("dustbin_loss", torch.tensor(0.0, device=device))
            dustbin_margin_loss = outputs.get("dustbin_margin_loss", torch.tensor(0.0, device=device))
            dustbin_margin_loss_weight = float(outputs.get("dustbin_margin_loss_weight", 0.0))
            

            total_loss = _compose_total_loss(
                primary_loss=loss,
                ks_loss=ks_loss,
                dustbin_loss=dustbin_loss,
                dustbin_margin_loss=dustbin_margin_loss,
                dustbin_margin_loss_weight=dustbin_margin_loss_weight,
            )

            if not torch.isfinite(total_loss).all():
                print(f"[WARN] Non-finite loss at iter {iter_num}, skipping batch.")
                logger.warning("Non-finite loss at iter %s, skipping batch.", iter_num)
                continue
            
            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
            dustbin_loss_value = dustbin_loss.item() if isinstance(dustbin_loss, torch.Tensor) else dustbin_loss
            dustbin_margin_loss_value = (
                dustbin_margin_loss.item()
                if isinstance(dustbin_margin_loss, torch.Tensor)
                else dustbin_margin_loss
            )
            total_loss_value = total_loss.item()
            total_loss.backward()
            grad_group_norms, grad_group_has_grad, grad_group_nonfinite = _group_grad_stats(model)
            for group_name, group_norm in grad_group_norms.items():
                if grad_group_nonfinite[group_name]:
                    logger.warning(
                        "Non-finite gradients detected for %s at epoch %s iter %s.",
                        group_name,
                        epoch,
                        iter_num,
                    )
                if grad_group_has_grad[group_name]:
                    grad_norm_accum[group_name] += float(group_norm)
                    grad_norm_count[group_name] += 1
                    if group_norm < 1e-8:
                        zero_grad_streak[group_name] += 1
                        if zero_grad_streak[group_name] % 20 == 0:
                            logger.warning(
                                "Near-zero gradient norm for %s has persisted for %s steps (epoch %s).",
                                group_name,
                                zero_grad_streak[group_name],
                                epoch,
                            )
                    else:
                        zero_grad_streak[group_name] = 0

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            # Adjust Learning weights
            optimizer.step()
            if optimizer_k is not None:
                optimizer_k.step()

            # Update running sums
            epoch_loss_sum += loss_value
            epoch_total_loss_sum += total_loss_value
            running_ks_loss += ks_loss_value
            running_dustbin_loss += dustbin_loss_value
            running_dustbin_margin_loss += dustbin_margin_loss_value
            running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error

            k_mae = k_rmse = k_zero_acc = None
            if "k_pred_count" in outputs and "gt_ks" in outputs:
                k_pred = outputs["k_pred_count"].detach().view(-1)
                k_gt = outputs["gt_ks"].detach().view(-1)
                k_mae = torch.mean(torch.abs(k_pred - k_gt)).item()
                k_rmse = torch.sqrt(torch.mean((k_pred - k_gt) ** 2)).item()
                if "label" in batch:
                    labels = batch["label"].to(device).view(-1)
                    imp_mask = labels < 0.5
                    if torch.any(imp_mask):
                        k_zero_acc = (k_pred[imp_mask] < 0.5).float().mean().item()

        
            # Report accuracy
            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()
            epoch_accuracy_sum += acc

            global_step = (epoch - start_epoch) * steps_per_epoch + batch_idx
            if iter_num % 5 == 0:
                writer.add_scalar('Train/Loss_Batch', loss_value, global_step)
                writer.add_scalar('Train/KS_Loss_Batch', ks_loss_value, global_step)
                writer.add_scalar('Train/Dustbin_Loss_Batch', dustbin_loss_value, global_step)
                writer.add_scalar('Train/Dustbin_Margin_Loss_Batch', dustbin_margin_loss_value, global_step)
                writer.add_scalar('Train/Total_Loss_Batch', total_loss_value, global_step)
                writer.add_scalar('Train/Accuracy_Batch', acc, global_step)
                writer.add_scalar('Train/Grad_Norm', float(grad_norm), global_step)
                writer.add_scalar('GradNorm/Matcher', grad_group_norms["Matcher"], global_step)
                writer.add_scalar('GradNorm/KHead', grad_group_norms["KHead"], global_step)
                writer.add_scalar('GradNorm/Dustbin', grad_group_norms["Dustbin"], global_step)
                if k_mae is not None:
                    writer.add_scalar('Train/K_MAE_Count', k_mae, global_step)
                if k_rmse is not None:
                    writer.add_scalar('Train/K_RMSE_Count', k_rmse, global_step)
                if k_zero_acc is not None:
                    writer.add_scalar('Train/K_Zero_Acc', k_zero_acc, global_step)
                if "dustbin_margin_value" in outputs:
                    writer.add_scalar('Train/Dustbin_Reject_Margin', float(outputs["dustbin_margin_value"]), global_step)

                avg_loss = epoch_loss_sum / iter_num
                avg_ks_loss = running_ks_loss / iter_num
                avg_dustbin_loss = running_dustbin_loss / iter_num
                avg_dustbin_margin_loss = running_dustbin_margin_loss / iter_num
                avg_total_loss = epoch_total_loss_sum / iter_num

                if "ks_loss" in outputs and optimizer_k is not None:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                            f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                            f"dustbin_loss: {avg_dustbin_loss:.4f}, "
                            f"dustbin_margin: {avg_dustbin_margin_loss:.4f}, "
                            f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}")
                else:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                            f"Loss: {avg_loss:.4f}, dustbin_loss: {avg_dustbin_loss:.4f}, "
                            f"dustbin_margin: {avg_dustbin_margin_loss:.4f}, "
                            f"Acc: {acc:.4f}")
                print(log_msg)
                logger.info(log_msg)

    avg_epoch_loss = epoch_loss_sum / iter_num
    avg_ks_loss = running_ks_loss / iter_num
    avg_dustbin_loss = running_dustbin_loss / iter_num
    avg_dustbin_margin_loss = running_dustbin_margin_loss / iter_num
    avg_total_loss = epoch_total_loss_sum / iter_num
    avg_accuracy = epoch_accuracy_sum / iter_num



    writer.add_scalar('Train/Loss_Epoch', avg_epoch_loss, epoch)
    writer.add_scalar('Train/KS_Loss_Epoch', avg_ks_loss, epoch)
    writer.add_scalar('Train/Dustbin_Loss_Epoch', avg_dustbin_loss, epoch)
    writer.add_scalar('Train/Dustbin_Margin_Loss_Epoch', avg_dustbin_margin_loss, epoch)
    writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
    writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)
    writer.add_scalar(
        'GradNormEpoch/Matcher',
        grad_norm_accum["Matcher"] / max(grad_norm_count["Matcher"], 1),
        epoch,
    )
    writer.add_scalar(
        'GradNormEpoch/KHead',
        grad_norm_accum["KHead"] / max(grad_norm_count["KHead"], 1),
        epoch,
    )
    writer.add_scalar(
        'GradNormEpoch/Dustbin',
        grad_norm_accum["Dustbin"] / max(grad_norm_count["Dustbin"], 1),
        epoch,
    )

    log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
              f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Dustbin Loss: {avg_dustbin_loss:.4f}, "
              f"Avg Dustbin Margin Loss: {avg_dustbin_margin_loss:.4f}, "
              f"Avg Total Loss: {avg_total_loss:.4f}, ")
    print(log_msg)
    logger.info(log_msg)

    # Save model and optimizer states
    save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))


    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
