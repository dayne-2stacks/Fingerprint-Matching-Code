import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from itertools import islice
from utils.data_to_cuda import data_to_cuda
from src.evaluation_metric import matching_accuracy
from utils.visualize import to_grayscale_cv2_image, visualize_match
from utils.matching import build_matches
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


def validate_epoch(model, dataloader, criterion, device, writer, epoch, logger, stage=None, max_iters=None):
    # Set model to evaluation mode
    model.eval()

    # Initialize running sums and counters
    val_loss_sum = 0.0
    val_ks_sum = 0.0
    val_dustbin_sum = 0.0
    val_dustbin_margin_sum = 0.0
    val_total_sum = 0.0
    val_num = 0
    val_accuracy_sum = 0.0
    k_pred_list = []
    k_gt_list = []
    k_zero_hits = 0
    k_zero_total = 0


    with torch.no_grad():
        for batch in islice(dataloader, max_iters):
            val_num += 1

            # Send data to device
            batch = data_to_cuda(batch)

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

            # Compute loss
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            dustbin_loss = outputs.get("dustbin_loss", torch.tensor(0.0, device=device))
            dustbin_margin_loss = outputs.get("dustbin_margin_loss", torch.tensor(0.0, device=device))
            dustbin_margin_loss_weight = float(outputs.get("dustbin_margin_loss_weight", 0.0))
            if "k_pred_count" in outputs and "gt_ks" in outputs:
                k_pred_list.append(outputs["k_pred_count"].detach().view(-1).cpu())
                k_gt_list.append(outputs["gt_ks"].detach().view(-1).cpu())
                if "label" in batch:
                    labels = batch["label"].detach().view(-1).cpu()
                    imp_mask = labels < 0.5
                    if torch.any(imp_mask):
                        k_zero_hits += (outputs["k_pred_count"].detach().view(-1).cpu()[imp_mask] < 0.5).sum().item()
                        k_zero_total += int(imp_mask.sum().item())
            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else float(ks_loss)
            dustbin_loss_value = dustbin_loss.item() if isinstance(dustbin_loss, torch.Tensor) else float(dustbin_loss)
            dustbin_margin_loss_value = (
                dustbin_margin_loss.item()
                if isinstance(dustbin_margin_loss, torch.Tensor)
                else float(dustbin_margin_loss)
            )
            total_loss_value = (
                loss_value
                + ks_loss_value
                + dustbin_loss_value
                + (dustbin_margin_loss_value * dustbin_margin_loss_weight)
            )
            
            # Report accuracy
            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()

            val_accuracy_sum += acc
            val_loss_sum += loss_value
            val_ks_sum += ks_loss_value
            val_dustbin_sum += dustbin_loss_value
            val_dustbin_margin_sum += dustbin_margin_loss_value
            val_total_sum += total_loss_value

            if val_num % 5 == 0:
                print(f"Validation batch {val_num} - Loss: {loss_value:.4f}, KS Loss: {ks_loss_value:.4f}, Total Loss: {total_loss_value:.4f}")

    avg_val_loss = val_loss_sum / val_num
    avg_ks_loss = val_ks_sum / val_num
    avg_dustbin_loss = val_dustbin_sum / val_num
    avg_dustbin_margin = val_dustbin_margin_sum / val_num
    avg_val_total = val_total_sum / val_num
    avg_val_accuracy = val_accuracy_sum / val_num
    if k_pred_list and k_gt_list:
        k_pred_all = torch.cat(k_pred_list)
        k_gt_all = torch.cat(k_gt_list)
        k_mae = torch.mean(torch.abs(k_pred_all - k_gt_all)).item()
        k_rmse = torch.sqrt(torch.mean((k_pred_all - k_gt_all) ** 2)).item()
        writer.add_scalar('Validation/K_MAE_Count', k_mae, epoch)
        writer.add_scalar('Validation/K_RMSE_Count', k_rmse, epoch)
        if k_zero_total > 0:
            writer.add_scalar('Validation/K_Zero_Acc', k_zero_hits / k_zero_total, epoch)



    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/KS_Loss', avg_ks_loss, epoch)
    writer.add_scalar('Validation/Dustbin_Loss', avg_dustbin_loss, epoch)
    writer.add_scalar('Validation/Dustbin_Margin_Loss', avg_dustbin_margin, epoch)
    writer.add_scalar('Validation/Total_Loss', avg_val_total, epoch)
    writer.add_scalar('Validation/Accuracy', avg_val_accuracy, epoch)
    
    log_msg = (
        f"Epoch {epoch} Validation: Primary Loss = {avg_val_loss:.4f}, "
        f"KS Loss = {avg_ks_loss:.4f}, Dustbin Loss = {avg_dustbin_loss:.4f}, "
        f"Dustbin Margin Loss = {avg_dustbin_margin:.4f}, "
        f"Total Loss = {avg_val_total:.4f}, "
    )
    print(log_msg)
    logger.info(log_msg)

    return avg_val_loss, avg_ks_loss, avg_val_total, avg_val_accuracy

def test_evaluation(model, dataloader, criterion, device, writer, epoch, stage=None, max_iters=None):
    model.eval()
    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    test_num = 0
    last_batch = None
    last_outputs = None
    genuine_pair = None
    imposter_pair = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(islice(dataloader, max_iters)):
            test_num += 1
            batch = data_to_cuda(batch)
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
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()

           
            test_loss_sum += loss.item()
            test_accuracy_sum += acc

            if stage is not None and stage >= 2 and 'label' in batch:
                # Handle batch of labels instead of assuming single element
                labels = batch['label'] if isinstance(batch['label'], torch.Tensor) else torch.tensor(batch['label'])
                
                # Process each sample in the batch
                for i, lbl in enumerate(labels):
                    lbl_val = lbl.item() if isinstance(lbl, torch.Tensor) else float(lbl)
                    
                    if lbl_val == 1 and genuine_pair is None:
                        # Extract single sample from batch for genuine pair
                        single_batch = {}
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor) and v.dim() > 0:
                                single_batch[k] = v[i:i+1]
                            elif isinstance(v, list):
                                single_batch[k] = [x[i:i+1] if isinstance(x, torch.Tensor) else x for x in v]
                            else:
                                single_batch[k] = v

                        single_outputs = {}
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor) and v.dim() > 0:
                                single_outputs[k] = v[i:i+1]
                            elif isinstance(v, list):
                                single_outputs[k] = [x[i:i+1] if isinstance(x, torch.Tensor) else x for x in v]
                            else:
                                single_outputs[k] = v

                        genuine_pair = (single_batch, single_outputs)
                    elif lbl_val == 0 and imposter_pair is None:
                        # Extract single sample from batch for imposter pair
                        single_batch = {}
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor) and v.dim() > 0:
                                single_batch[k] = v[i:i+1]
                            elif isinstance(v, list):
                                single_batch[k] = [x[i:i+1] if isinstance(x, torch.Tensor) else x for x in v]
                            else:
                                single_batch[k] = v

                        single_outputs = {}
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor) and v.dim() > 0:
                                single_outputs[k] = v[i:i+1]
                            elif isinstance(v, list):
                                single_outputs[k] = [x[i:i+1] if isinstance(x, torch.Tensor) else x for x in v]
                            else:
                                single_outputs[k] = v

                        imposter_pair = (single_batch, single_outputs)

                    if genuine_pair is not None and imposter_pair is not None:
                        break
                
                if genuine_pair is not None and imposter_pair is not None:
                    continue

            if batch_idx == 0:
                last_batch = batch
                last_outputs = outputs

    denom = max(test_num, 1)
    avg_test_loss = test_loss_sum / denom
    avg_test_accuracy = test_accuracy_sum / denom

    writer.add_scalar('Test/Loss', avg_test_loss, epoch)
    writer.add_scalar('Test/Accuracy', avg_test_accuracy, epoch)

    def _visualize(batch, outputs, tag):
        if 'Ps' in batch:
            kp0 = batch['Ps'][0][0].cpu().numpy()
            kp1 = batch['Ps'][1][0].cpu().numpy()
        else:
            kp0 = np.array([[100, 100], [150, 150], [200, 200]])
            kp1 = np.array([[110, 110], [160, 160], [210, 210]])

        ds_mat = outputs["ds_mat"].cpu().numpy()[0]
        per_mat = outputs["perm_mat"].cpu().numpy()[0]
        n1 = outputs["ns"][0]
        n2 = outputs["ns"][1]
        if isinstance(n1, torch.Tensor):
            n1 = int(n1[0].item()) if n1.dim() > 0 else int(n1.item())
        else:
            n1 = int(n1)
        if isinstance(n2, torch.Tensor):
            n2 = int(n2[0].item()) if n2.dim() > 0 else int(n2.item())
        else:
            n2 = int(n2)
        kp0 = kp0[:n1]
        kp1 = kp1[:n2]
        ds_mat = ds_mat[:n1, :n2]
        per_mat = per_mat[:n1, :n2]
        matches = build_matches(ds_mat, per_mat)

        if "id_list" in batch:
            img0 = batch["images"][0][0]
            img1 = batch["images"][1][0]
        else:
            img0 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg")
            img1 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg")

        img0 = to_grayscale_cv2_image(img0)
        img1 = to_grayscale_cv2_image(img1)

        match_path = f"photos/test_photos/{tag}_{epoch}.jpg"
        visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/test_photos/", filename=f"{tag}_{epoch}.jpg")

        if os.path.exists(match_path):
            match_img = cv2.imread(match_path)
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            writer.add_image(f'Test/{tag.capitalize()}', match_img.transpose(2, 0, 1), epoch, dataformats='CHW')

    if last_batch is not None and last_outputs is not None:
        _visualize(last_batch, last_outputs, 'match')

    if stage is not None and stage >= 2:
        if genuine_pair is not None:
            _visualize(genuine_pair[0], genuine_pair[1], 'genuine_match')
        if imposter_pair is not None:
            _visualize(imposter_pair[0], imposter_pair[1], 'imposter_match')

    print(f"Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, Test Accuracy = {avg_test_accuracy:.4f}")
    return avg_test_loss, avg_test_accuracy
