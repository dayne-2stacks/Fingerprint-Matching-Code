import os
import cv2
import numpy as np
import torch
from utils.data_to_cuda import data_to_cuda
from src.evaluation_metric import matching_accuracy
from utils.visualize import to_grayscale_cv2_image, visualize_match
from utils.matching import build_matches


def validate_epoch(model, dataloader, criterion, device, writer, epoch, logger):
    model.eval()
    val_loss_sum = 0.0
    val_ks_sum = 0.0
    val_cls_sum = 0.0
    val_total_sum = 0.0
    val_num = 0
    val_accuracy_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            val_num += 1
            batch = data_to_cuda(batch)
            outputs = model(batch)
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            cls_loss = outputs.get("cls_loss", torch.tensor(0.0, device=device))

            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else float(ks_loss)
            cls_loss_value = cls_loss.item() if isinstance(cls_loss, torch.Tensor) else float(cls_loss)
            total_loss_value = loss_value + ks_loss_value + cls_loss_value

            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()

            val_accuracy_sum += acc
            val_loss_sum += loss_value
            val_ks_sum += ks_loss_value
            val_cls_sum += cls_loss_value
            val_total_sum += total_loss_value

            if val_num % 5 == 0:
                print(f"Validation batch {val_num} - Loss: {loss_value:.4f}, KS Loss: {ks_loss_value:.4f}, Total Loss: {total_loss_value:.4f}")

    avg_val_loss = val_loss_sum / len(dataloader)
    avg_ks_loss = val_ks_sum / len(dataloader)
    avg_val_total = val_total_sum / len(dataloader)
    avg_val_accuracy = val_accuracy_sum / len(dataloader)
    avg_cls_loss = val_cls_sum / len(dataloader)

    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/KS_Loss', avg_ks_loss, epoch)
    writer.add_scalar('Validation/Cls_Loss', avg_cls_loss, epoch)
    writer.add_scalar('Validation/Total_Loss', avg_val_total, epoch)
    writer.add_scalar('Validation/Accuracy', avg_val_accuracy, epoch)

    log_msg = f"Epoch {epoch} Validation: Primary Loss = {avg_val_loss:.4f}, KS Loss = {avg_ks_loss:.4f}, CLS Loss = {avg_cls_loss:.4f}, Total Loss = {avg_val_total:.4f}"
    print(log_msg)
    logger.info(log_msg)

    return avg_val_loss, avg_ks_loss, avg_val_total, avg_val_accuracy


def test_evaluation(model, dataloader, criterion, device, writer, epoch, stage=None):
    model.eval()
    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    test_cls_sum = 0.0
    last_batch = None
    last_outputs = None
    genuine_pair = None
    imposter_pair = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = data_to_cuda(batch)
            outputs = model(batch)
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            cls_loss = outputs.get("cls_loss", torch.tensor(0.0, device=device))
            acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
            if isinstance(acc, torch.Tensor):
                if acc.numel() > 1:
                    acc = acc.mean().item()
                else:
                    acc = acc.item()

            test_loss_sum += loss.item() + (cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss)
            test_cls_sum += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
            test_accuracy_sum += acc

            if stage == 4 and 'label' in batch:
                lbl = batch['label'].item() if isinstance(batch['label'], torch.Tensor) else float(batch['label'])
                if lbl == 1 and genuine_pair is None:
                    genuine_pair = (batch, outputs)
                elif lbl == 0 and imposter_pair is None:
                    imposter_pair = (batch, outputs)

                if genuine_pair is not None and imposter_pair is not None:
                    continue

            if batch_idx == 0:
                last_batch = batch
                last_outputs = outputs

    avg_test_loss = test_loss_sum / len(dataloader)
    avg_test_accuracy = test_accuracy_sum / len(dataloader)
    avg_test_cls = test_cls_sum / len(dataloader)

    writer.add_scalar('Test/Loss', avg_test_loss, epoch)
    writer.add_scalar('Test/Cls_Loss', avg_test_cls, epoch)
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

    if stage == 4:
        if genuine_pair is not None:
            _visualize(genuine_pair[0], genuine_pair[1], 'genuine_match')
        if imposter_pair is not None:
            _visualize(imposter_pair[0], imposter_pair[1], 'imposter_match')

    print(f"Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, CLS Loss = {avg_test_cls:.4f}, Test Accuracy = {avg_test_accuracy:.4f}")
    return avg_test_loss, avg_test_accuracy
