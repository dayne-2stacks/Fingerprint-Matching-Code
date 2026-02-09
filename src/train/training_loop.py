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



def train_epoch(model, dataloader, criterion, optimizer, optimizer_k,
                device, writer, epoch, start_epoch, stage, logger, checkpoint_path,
                sg_dustbin_weight=1.0, auth_weight=1.0, stage2_full_loss=False,
                detect_anomaly=False, max_iters=None):
    # Initialize running sums and counters
    epoch_loss_sum = 0.0
    running_ks_loss = 0.0
    running_dustbin_loss = 0.0
    running_sg_dustbin_loss = 0.0
    running_k_reg_loss = 0.0
    running_k_cls_loss = 0.0
    running_ks_error = 0.0
    running_auth_loss = 0.0
    epoch_accuracy_sum = 0.0
    epoch_total_loss_sum = 0.0
    iter_num = 0

    torch.autograd.set_detect_anomaly(bool(detect_anomaly))
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
        k_reg_loss = outputs.get("k_reg_loss", torch.tensor(0.0, device=device))
        k_cls_loss = outputs.get("k_cls_loss", torch.tensor(0.0, device=device))
        dustbin_loss = outputs.get("dustbin_loss", torch.tensor(0.0, device=device))
        sg_dustbin_loss = outputs.get("sg_dustbin_loss", torch.tensor(0.0, device=device))
        auth_loss = torch.tensor(0.0, device=device)
        if stage == 4 and "label" in batch and "auth_logit" in outputs:
            labels = batch["label"].to(device).view(-1).float()
            auth_loss = F.binary_cross_entropy_with_logits(
                outputs["auth_logit"].view(-1), labels
            )

        total_loss = (
            loss
            + (ks_loss if isinstance(ks_loss, torch.Tensor) else torch.tensor(ks_loss, device=device))
            + (dustbin_loss if isinstance(dustbin_loss, torch.Tensor) else torch.tensor(dustbin_loss, device=device))
            + (sg_dustbin_loss if isinstance(sg_dustbin_loss, torch.Tensor) else torch.tensor(sg_dustbin_loss, device=device)) * sg_dustbin_weight
            + auth_loss * auth_weight
        )
        if stage == 2 and not stage2_full_loss:
            total_loss = ks_loss if isinstance(ks_loss, torch.Tensor) else torch.tensor(ks_loss, device=device)

        if not torch.isfinite(total_loss).all():
            print(f"[WARN] Non-finite loss at iter {iter_num}, skipping batch.")
            logger.warning("Non-finite loss at iter %s, skipping batch.", iter_num)
            continue
        
        loss_value = loss.item()
        ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
        k_reg_loss_value = k_reg_loss.item() if isinstance(k_reg_loss, torch.Tensor) else k_reg_loss
        k_cls_loss_value = k_cls_loss.item() if isinstance(k_cls_loss, torch.Tensor) else k_cls_loss
        dustbin_loss_value = dustbin_loss.item() if isinstance(dustbin_loss, torch.Tensor) else dustbin_loss
        sg_dustbin_loss_value = sg_dustbin_loss.item() if isinstance(sg_dustbin_loss, torch.Tensor) else sg_dustbin_loss
        total_loss_value = total_loss.item()
        total_loss.backward()
        if iter_num % 2 == 0 and hasattr(model, "bin_score"):
            bin_grad = getattr(model.bin_score, "grad", None)
            print(
                f"bin_score grad: {None if bin_grad is None else bin_grad.detach().cpu().item():}"
            )
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
        running_sg_dustbin_loss += sg_dustbin_loss_value
        running_k_reg_loss += k_reg_loss_value
        running_k_cls_loss += k_cls_loss_value
        running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error
        running_auth_loss += auth_loss.item() if isinstance(auth_loss, torch.Tensor) else auth_loss

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
            writer.add_scalar('Train/SG_Dustbin_Loss_Batch', sg_dustbin_loss_value, global_step)
            writer.add_scalar('Train/K_Reg_Loss_Batch', k_reg_loss_value, global_step)
            writer.add_scalar('Train/K_Cls_Loss_Batch', k_cls_loss_value, global_step)
            writer.add_scalar('Train/Auth_Loss_Batch', auth_loss.item(), global_step)
            writer.add_scalar('Train/Total_Loss_Batch', total_loss_value, global_step)
            writer.add_scalar('Train/Accuracy_Batch', acc, global_step)
            writer.add_scalar('Train/Grad_Norm', float(grad_norm), global_step)
            if k_mae is not None:
                writer.add_scalar('Train/K_MAE_Count', k_mae, global_step)
            if k_rmse is not None:
                writer.add_scalar('Train/K_RMSE_Count', k_rmse, global_step)
            if k_zero_acc is not None:
                writer.add_scalar('Train/K_Zero_Acc', k_zero_acc, global_step)
            if "k_count_loss" in outputs:
                writer.add_scalar('Train/K_Count_Loss_Batch', float(outputs["k_count_loss"]), global_step)

            avg_loss = epoch_loss_sum / iter_num
            avg_ks_loss = running_ks_loss / iter_num
            avg_dustbin_loss = running_dustbin_loss / iter_num
            avg_sg_dustbin_loss = running_sg_dustbin_loss / iter_num
            avg_k_reg_loss = running_k_reg_loss / iter_num
            avg_k_cls_loss = running_k_cls_loss / iter_num
            avg_auth_loss = running_auth_loss / iter_num
            avg_total_loss = epoch_total_loss_sum / iter_num

            if "ks_loss" in outputs and optimizer_k is not None:
                log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                          f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                          f"k_reg: {avg_k_reg_loss:.4f}, k_cls: {avg_k_cls_loss:.4f}, "
                          f"auth_loss: {avg_auth_loss:.4f}, "
                          f"dustbin_loss: {avg_dustbin_loss:.4f}, sg_dustbin_loss: {avg_sg_dustbin_loss:.4f},  "
                          f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}")
            else:
                log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                          f"Loss: {avg_loss:.4f}, dustbin_loss: {avg_dustbin_loss:.4f}, sg_dustbin_loss: {avg_sg_dustbin_loss:.4f}, "
                          f"Acc: {acc:.4f}")
            print(log_msg)
            logger.info(log_msg)

    avg_epoch_loss = epoch_loss_sum / iter_num
    avg_ks_loss = running_ks_loss / iter_num
    avg_dustbin_loss = running_dustbin_loss / iter_num
    avg_sg_dustbin_loss = running_sg_dustbin_loss / iter_num
    avg_k_reg_loss = running_k_reg_loss / iter_num
    avg_k_cls_loss = running_k_cls_loss / iter_num
    avg_auth_loss = running_auth_loss / iter_num
    avg_total_loss = epoch_total_loss_sum / iter_num
    avg_accuracy = epoch_accuracy_sum / iter_num



    writer.add_scalar('Train/Loss_Epoch', avg_epoch_loss, epoch)
    writer.add_scalar('Train/KS_Loss_Epoch', avg_ks_loss, epoch)
    writer.add_scalar('Train/Dustbin_Loss_Epoch', avg_dustbin_loss, epoch)
    writer.add_scalar('Train/SG_Dustbin_Loss_Epoch', avg_sg_dustbin_loss, epoch)
    writer.add_scalar('Train/K_Reg_Loss_Epoch', avg_k_reg_loss, epoch)
    writer.add_scalar('Train/K_Cls_Loss_Epoch', avg_k_cls_loss, epoch)
    writer.add_scalar('Train/Auth_Loss_Epoch', avg_auth_loss, epoch)
    writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
    writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)

    log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
              f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Dustbin Loss: {avg_dustbin_loss:.4f}, "
              f"Avg SG Dustbin Loss: {avg_sg_dustbin_loss:.4f}, "
              f"Avg K Reg Loss: {avg_k_reg_loss:.4f}, Avg K Cls Loss: {avg_k_cls_loss:.4f}, "
              f"Avg Auth Loss: {avg_auth_loss:.4f}, "
              f"Avg Total Loss: {avg_total_loss:.4f}, ")
    print(log_msg)
    logger.info(log_msg)

    # Save model and optimizer states
    save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))


    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
