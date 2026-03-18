import os
import cv2
import numpy as np
import torch
from itertools import islice
from utils.data_to_cuda import data_to_cuda
from src.train.common import (
    batch_match_counts,
    compose_total_loss,
    counts_summary,
    summary_scalars,
)
from src.train.inspect_utils import slice_batch
from utils.visualize import to_grayscale_cv2_image, visualize_match
from utils.matching import build_matches
from src.model.dustbin import ns_pair_to_ints, strip_dustbin_from_outputs


def validate_epoch(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    logger,
    stage=None,
    max_iters=None,
):
    # Set model to evaluation mode
    model.eval()

    # Initialize running sums and counters
    val_loss_sum = 0.0
    val_ks_sum = 0.0
    val_total_sum = 0.0
    val_num = 0
    metric_tp_sum = 0.0
    metric_tn_sum = 0.0
    metric_fp_sum = 0.0
    metric_fn_sum = 0.0



    with torch.no_grad():
        for batch in islice(dataloader, max_iters):
            val_num += 1

            # Send data to device
            batch = data_to_cuda(batch)

             # outputs = model(batch, stage=stage,)
            if stage == 1:
                # Forward pass
                outputs = model(batch, regression=False)
            else:
                outputs = model(batch, regression=True)
            # if stage == 3:
            #     loss = criterion(outputs["ds_mat_dustbin"], outputs["gt_perm_mat"], *outputs["ns"])

            # else: 
            #     strip_dustbin_from_outputs(outputs)

            #     # compute loss and their gradients
            #     if stage == 1:
            #         loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])

            #     elif stage == 4:
            #         loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])

            #     elif stage == 2:
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else float(ks_loss)

           
            total_loss = compose_total_loss(
                loss,
                ks_loss,
                ms_loss=torch.tensor(0.0, device=device),
                stage=stage,
            )
            total_loss_value = float(total_loss.item())
            
            batch_counts = batch_match_counts(outputs)
            batch_tp = batch_counts["tp"]
            batch_tn = batch_counts["tn"]
            batch_fp = batch_counts["fp"]
            batch_fn = batch_counts["fn"]

            metric_tp_sum += batch_tp
            metric_tn_sum += batch_tn
            metric_fp_sum += batch_fp
            metric_fn_sum += batch_fn

            batch_scalars = summary_scalars(counts_summary(batch_tp, batch_tn, batch_fp, batch_fn, device))

            acc = batch_scalars["accuracy"]
            prec = batch_scalars["precision"]
            rec = batch_scalars["recall"]
            f1 = batch_scalars["f1"]
            micro_f1 = batch_scalars["micro_f1"]
            macro_f1 = batch_scalars["macro_f1"]

            val_loss_sum += loss_value
            val_ks_sum += ks_loss_value
            val_total_sum += total_loss_value

            if val_num % 5 == 0:
                msg = (
                    f"Validation batch {val_num} - Loss: {loss_value:.4f}, KS Loss: {ks_loss_value:.4f}, "
                    f"Total Loss: {total_loss_value:.4f}, "
                    f"Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, "
                    f"F1: {f1:.4f}, MiF1: {micro_f1:.4f}, MaF1: {macro_f1:.4f}"
                )


                print(msg)

    avg_val_loss = val_loss_sum / val_num
    avg_ks_loss = val_ks_sum / val_num

    avg_val_total = val_total_sum / val_num
    val_scalars = summary_scalars(counts_summary(metric_tp_sum, metric_tn_sum, metric_fp_sum, metric_fn_sum, device))
    avg_val_accuracy = val_scalars["accuracy"]
    avg_val_precision = val_scalars["precision"]
    avg_val_recall = val_scalars["recall"]
    avg_val_f1 = val_scalars["f1"]
    avg_val_micro_f1 = val_scalars["micro_f1"]
    avg_val_macro_f1 = val_scalars["macro_f1"]

    log_msg = (
        f"Epoch {epoch} Validation: Primary Loss = {avg_val_loss:.4f}, "
        f"KS Loss = {avg_ks_loss:.4f}, "
        f"Total Loss = {avg_val_total:.4f}, Acc = {avg_val_accuracy:.4f}, P = {avg_val_precision:.4f}, "
        f"R = {avg_val_recall:.4f}, F1 = {avg_val_f1:.4f}, MiF1 = {avg_val_micro_f1:.4f}, "
        f"MaF1 = {avg_val_macro_f1:.4f}"
    )
    print(log_msg)
    logger.info(log_msg)

    return avg_val_loss, avg_ks_loss, avg_val_total, avg_val_accuracy

def test_evaluation(model, dataloader, criterion, device, writer, epoch, stage=None, max_iters=None):
    model.eval()
    test_loss_sum = 0.0
    metric_tp_sum = 0.0
    metric_tn_sum = 0.0
    metric_fp_sum = 0.0
    metric_fn_sum = 0.0
    test_num = 0
    last_batch = None
    last_outputs = None
    genuine_pair = None
    imposter_pair = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(islice(dataloader, max_iters)):
            test_num += 1
            batch = data_to_cuda(batch)
            if stage is not None:
                batch["stage_id"] = int(stage)
            outputs = model(batch, stage=stage)
            strip_dustbin_from_outputs(outputs)
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            batch_counts = batch_match_counts(outputs)
            batch_tp = batch_counts["tp"]
            batch_tn = batch_counts["tn"]
            batch_fp = batch_counts["fp"]
            batch_fn = batch_counts["fn"]
            metric_tp_sum += batch_tp
            metric_tn_sum += batch_tn
            metric_fp_sum += batch_fp
            metric_fn_sum += batch_fn

           
            test_loss_sum += loss.item()

            if stage is not None and stage >= 2 and 'label' in batch:
                # Handle batch of labels instead of assuming single element
                labels = batch['label'] if isinstance(batch['label'], torch.Tensor) else torch.tensor(batch['label'])
                
                # Process each sample in the batch
                for i, lbl in enumerate(labels):
                    lbl_val = lbl.item() if isinstance(lbl, torch.Tensor) else float(lbl)
                    
                    if lbl_val == 1 and genuine_pair is None:
                        genuine_pair = (slice_batch(batch, i), slice_batch(outputs, i))
                    elif lbl_val == 0 and imposter_pair is None:
                        imposter_pair = (slice_batch(batch, i), slice_batch(outputs, i))

                    if genuine_pair is not None and imposter_pair is not None:
                        break
                
                if genuine_pair is not None and imposter_pair is not None:
                    continue

            if batch_idx == 0:
                last_batch = batch
                last_outputs = outputs

    denom = max(test_num, 1)
    avg_test_loss = test_loss_sum / denom
    test_scalars = summary_scalars(counts_summary(metric_tp_sum, metric_tn_sum, metric_fp_sum, metric_fn_sum, device))
    avg_test_accuracy = test_scalars["accuracy"]
    avg_test_precision = test_scalars["precision"]
    avg_test_recall = test_scalars["recall"]
    avg_test_f1 = test_scalars["f1"]
    avg_test_micro_f1 = test_scalars["micro_f1"]
    avg_test_macro_f1 = test_scalars["macro_f1"]



    def _visualize(batch, outputs, tag):
        if 'Ps' in batch:
            kp0 = batch['Ps'][0][0].cpu().numpy()
            kp1 = batch['Ps'][1][0].cpu().numpy()
        else:
            kp0 = np.array([[100, 100], [150, 150], [200, 200]])
            kp1 = np.array([[110, 110], [160, 160], [210, 210]])

        ds_mat = outputs["ds_mat"].cpu().numpy()[0]
        per_mat = outputs["perm_mat"].cpu().numpy()[0]
        n1, n2 = ns_pair_to_ints(outputs, sample_idx=0)
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
            _visualize(genuine_pair[0], genuine_pair[1], f'stage{stage}_genuine_match')
        if imposter_pair is not None:
            _visualize(imposter_pair[0], imposter_pair[1], f'stage{stage}_imposter_match')

    print(
        f"Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, Test Accuracy = {avg_test_accuracy:.4f}, "
        f"P = {avg_test_precision:.4f}, R = {avg_test_recall:.4f}, F1 = {avg_test_f1:.4f}, "
        f"MiF1 = {avg_test_micro_f1:.4f}, MaF1 = {avg_test_macro_f1:.4f}"
    )
    return avg_test_loss, avg_test_accuracy
