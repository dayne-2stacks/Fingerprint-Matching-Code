import os
import torch
import numpy as np
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import save_model, load_model
from src.evaluation_metric import matching_accuracy
from utils.visualize import to_grayscale_cv2_image, visualize_match
from utils.matching import build_matches

"""[2025-06-29] k=0 no-match update: training loop with optional verbose stats."""


def train_epoch(model, dataloader, criterion, optimizer, optimizer_k, device, writer, epoch, start_epoch, stage, logger, checkpoint_path, verbose_nomatch=False):
    epoch_loss_sum = 0.0
    epoch_total_loss_sum = 0.0
    running_ks_loss = 0.0
    running_ks_error = 0.0
    epoch_accuracy_sum = 0.0
    iter_num = 0

    for i in range(3):
        for batch_idx, batch in enumerate(dataloader):
            iter_num += 1
            batch = data_to_cuda(batch)

            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))
            k_pred = outputs.get("k_pred", torch.zeros_like(ks_loss))

            if verbose_nomatch:
                nomask = (outputs["gt_perm_mat"].view(outputs["gt_perm_mat"].shape[0], -1).sum(dim=1) == 0)
                num_no = int(nomask.sum().item())
                mean_k = (k_pred[nomask].mean().item() if num_no else 0.0)
                reasons = []
                for idx_t in torch.nonzero(nomask).view(-1)[:3]:
                    shared = int(outputs['gt_perm_mat'][idx_t].sum().item())
                    if shared < 10:
                        reasons.append("<10 shared minutiae")
                    else:
                        reasons.append("no overlap")
                msg = f"no-match:{num_no} mean_k={mean_k:.3f} reasons: {'; '.join(reasons)}"
                print(msg)
                logger.info(msg)

            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
            total_loss = loss + (ks_loss if isinstance(ks_loss, torch.Tensor) else ks_loss)
            total_loss_value = total_loss.item()

            epoch_loss_sum += loss_value
            epoch_total_loss_sum += total_loss_value
            running_ks_loss += ks_loss_value
            running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error

            total_loss.backward()
            if stage == 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if optimizer_k is not None:
                optimizer_k.step()

            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()
            epoch_accuracy_sum += acc

            global_step = (epoch - start_epoch) * len(dataloader) * 3 + i * len(dataloader) + batch_idx
            if iter_num % 5 == 0:
                writer.add_scalar('Train/Loss_Batch', loss_value, global_step)
                writer.add_scalar('Train/KS_Loss_Batch', ks_loss_value, global_step)
                writer.add_scalar('Train/Total_Loss_Batch', total_loss_value, global_step)
                writer.add_scalar('Train/Accuracy_Batch', acc, global_step)

            if iter_num % 5 == 0:
                avg_loss = epoch_loss_sum / iter_num
                avg_ks_loss = running_ks_loss / iter_num
                avg_total_loss = epoch_total_loss_sum / iter_num

                if "ks_loss" in outputs and optimizer_k is not None:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                              f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                              f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}")
                else:
                    log_msg = f"Epoch: {epoch}, Iter: {iter_num}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}"
                print(log_msg)
                logger.info(log_msg)

    avg_epoch_loss = epoch_loss_sum / iter_num
    avg_ks_loss = running_ks_loss / iter_num
    avg_total_loss = epoch_total_loss_sum / iter_num
    avg_accuracy = epoch_accuracy_sum / iter_num

    writer.add_scalar('Train/Loss_Epoch', avg_epoch_loss, epoch)
    writer.add_scalar('Train/KS_Loss_Epoch', avg_ks_loss, epoch)
    writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
    writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)

    log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
              f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}")
    print(log_msg)
    logger.info(log_msg)

    save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))

    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
