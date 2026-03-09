import torch
from itertools import islice
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import save_model
from src.model.dustbin import strip_dustbin_from_outputs
from src.train.common import (
    batch_match_counts,
    compose_total_loss,
    counts_summary,
    k_debug_scalars,
    matching_metrics_from_counts,
    summary_scalars,
)


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


def train_epoch(model, dataloader, criterion, optimizer, optimizer_k,
                device, writer, epoch, start_epoch, stage, logger, checkpoint_path,
                detect_anomaly=False, max_iters=None, stage1_ss_base_aux_weight: float = 0.25):
    # Initialize running sums and counters
    epoch_loss_sum = 0.0
    running_ks_loss = 0.0
    running_dustbin_loss = 0.0
    running_ks_error = 0.0
    running_dustbin_k_mse = 0.0
    running_dustbin_bce = 0.0
    running_dustbin_k_mae = 0.0
    running_dustbin_k_balance = 0.0
    running_stage1_ss_base_aux = 0.0
    running_stage1_ss_base_aux_weighted = 0.0
    epoch_total_loss_sum = 0.0
    metric_tp_sum = 0.0
    metric_tn_sum = 0.0
    metric_fp_sum = 0.0
    metric_fn_sum = 0.0
    k_debug_weight_sum = 0.0
    k_debug_sums = {}
    iter_num = 0
    stage1_grad_diag_printed = False
    stage1_strip_diag_printed = False


    torch.autograd.set_detect_anomaly(bool(detect_anomaly))
    
    steps_per_epoch = max_iters if max_iters is not None else len(dataloader)
    # One dataloader pass per epoch; extra passes distort LR schedule and early stopping.
    for _ in range(2):
        for batch_idx, batch in enumerate(islice(dataloader, max_iters)):
            iter_num += 1

            batch = data_to_cuda(batch)
            if stage is not None:
                batch["stage_id"] = int(stage)

            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()

            outputs = model(batch)

            if stage == 1 and not stage1_strip_diag_printed:
                try:
                    gt_before = outputs.get("gt_perm_mat")
                    ns_before = outputs.get("ns")

                    def _ns_lists(ns_val):
                        if not isinstance(ns_val, (list, tuple)) or len(ns_val) < 2:
                            return [], []
                        src_ns, tgt_ns = ns_val[0], ns_val[1]
                        if isinstance(src_ns, torch.Tensor):
                            src_list = [int(x.item()) for x in src_ns.reshape(-1)]
                        else:
                            src_list = [int(x) for x in src_ns]
                        if isinstance(tgt_ns, torch.Tensor):
                            tgt_list = [int(x.item()) for x in tgt_ns.reshape(-1)]
                        else:
                            tgt_list = [int(x) for x in tgt_ns]
                        return src_list, tgt_list

                    src_before, tgt_before = _ns_lists(ns_before)
                    gt_shape_before = tuple(gt_before.shape) if isinstance(gt_before, torch.Tensor) else None
                    gt_sums_before = []
                    gt_before_ref = gt_before.detach().clone() if isinstance(gt_before, torch.Tensor) else None
                    if isinstance(gt_before, torch.Tensor) and src_before and tgt_before:
                        for b in range(min(gt_before.shape[0], 3)):
                            n1b = max(int(src_before[b]), 0)
                            n2b = max(int(tgt_before[b]), 0)
                            gt_sums_before.append(float(gt_before[b, :n1b, :n2b].sum().item()))
                except Exception as exc:
                    print(f"[Stage1 StripDiag] pre-strip capture failed: {exc}")
                    logger.warning("[Stage1 StripDiag] pre-strip capture failed: %s", exc)
                    gt_shape_before = None
                    gt_sums_before = []
                    src_before, tgt_before = [], []
                    gt_before_ref = None

            strip_dustbin_from_outputs(outputs)

            if stage == 1 and not stage1_strip_diag_printed:
                try:
                    gt_after = outputs.get("gt_perm_mat")
                    ns_after = outputs.get("ns")
                    src_after, tgt_after = _ns_lists(ns_after)
                    gt_shape_after = tuple(gt_after.shape) if isinstance(gt_after, torch.Tensor) else None
                    gt_sums_after = []
                    gt_real_sums_before = []
                    gt_real_sums_after = []
                    gt_real_max_abs_diff = []
                    if isinstance(gt_after, torch.Tensor) and src_after and tgt_after:
                        for b in range(min(gt_after.shape[0], 3)):
                            n1b = max(int(src_after[b]), 0)
                            n2b = max(int(tgt_after[b]), 0)
                            gt_sums_after.append(float(gt_after[b, :n1b, :n2b].sum().item()))
                            if isinstance(gt_before_ref, torch.Tensor):
                                before_real = gt_before_ref[b, :n1b, :n2b]
                                after_real = gt_after[b, :n1b, :n2b]
                                gt_real_sums_before.append(float(before_real.sum().item()))
                                gt_real_sums_after.append(float(after_real.sum().item()))
                                if before_real.numel() > 0:
                                    gt_real_max_abs_diff.append(float((before_real - after_real).abs().max().item()))
                                else:
                                    gt_real_max_abs_diff.append(0.0)

                    strip_msg = (
                        "[Stage1 StripDiag] "
                        f"gt_shape_before={gt_shape_before} "
                        f"gt_shape_after={gt_shape_after} "
                        f"ns_before_src={src_before[:3]} ns_before_tgt={tgt_before[:3]} "
                        f"ns_after_src={src_after[:3]} ns_after_tgt={tgt_after[:3]} "
                        f"gt_active_sum_before(first3)={gt_sums_before} "
                        f"gt_active_sum_after(first3)={gt_sums_after} "
                        f"gt_real_block_sum_before(first3)={gt_real_sums_before} "
                        f"gt_real_block_sum_after(first3)={gt_real_sums_after} "
                        f"gt_real_block_max_abs_diff(first3)={gt_real_max_abs_diff} "
                        f"has_dustbin={outputs.get('has_dustbin', None)}"
                    )
                    print(strip_msg)
                    logger.info(strip_msg)
                except Exception as exc:
                    warn_msg = f"[Stage1 StripDiag] post-strip capture failed: {exc}"
                    print(warn_msg)
                    logger.warning(warn_msg)
                stage1_strip_diag_printed = True

            batch_k_debug = k_debug_scalars(outputs)
            if batch_k_debug:
                batch_k_weight = float(batch_k_debug.get("sample_count", 1.0))
                if batch_k_weight > 0:
                    k_debug_weight_sum += batch_k_weight
                    for key, value in batch_k_debug.items():
                        if key == "sample_count":
                            continue
                        k_debug_sums[key] = k_debug_sums.get(key, 0.0) + (float(value) * batch_k_weight)

            # compute loss and their gradients
            if stage == 1:
                loss = criterion(outputs["ds_mat_topk"], outputs["gt_perm_mat"], *outputs["ns"])
                stage1_ss_base_aux_loss = criterion(outputs["ds_mat_matcher"], outputs["gt_perm_mat"], *outputs["ns"])
            elif stage == 4:
                loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
                stage1_ss_base_aux_loss = torch.tensor(0.0, device=device)
            elif stage == 2:
                loss = criterion(outputs["ds_mat_topk"], outputs["gt_perm_mat"], *outputs["ns"])
                stage1_ss_base_aux_loss = torch.tensor(0.0, device=device)
            elif stage == 3:
                loss = criterion(outputs["ds_mat_dustbin"], outputs["gt_perm_mat"], *outputs["ns"])
                stage1_ss_base_aux_loss = torch.tensor(0.0, device=device)
            else:
                loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
                stage1_ss_base_aux_loss = torch.tensor(0.0, device=device)
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))
            dustbin_loss = outputs.get("dustbin_loss", torch.tensor(0.0, device=device))
            dustbin_k_mse_loss = outputs.get("dustbin_k_mse_loss", torch.tensor(0.0, device=device))
            dustbin_bce_loss = outputs.get("dustbin_bce_loss", torch.tensor(0.0, device=device))
            dustbin_k_mae = outputs.get("dustbin_k_mae", torch.tensor(0.0, device=device))
            dustbin_k_balance_err = outputs.get("dustbin_k_balance_err", torch.tensor(0.0, device=device))

            total_loss = compose_total_loss(
                primary_loss=loss,
                ks_loss=ks_loss,
                dustbin_loss=dustbin_loss,
                stage=stage,
                stage1_ss_base_aux_loss=stage1_ss_base_aux_loss,
                stage1_ss_base_aux_weight=stage1_ss_base_aux_weight,
            )

            if not torch.isfinite(total_loss).all():
                print(f"[WARN] Non-finite loss at iter {iter_num}, skipping batch.")
                logger.warning("Non-finite loss at iter %s, skipping batch.", iter_num)
                continue

            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
            dustbin_loss_value = dustbin_loss.item() if isinstance(dustbin_loss, torch.Tensor) else dustbin_loss
            dustbin_k_mse_value = (
                dustbin_k_mse_loss.item() if isinstance(dustbin_k_mse_loss, torch.Tensor) else dustbin_k_mse_loss
            )
            dustbin_bce_value = (
                dustbin_bce_loss.item() if isinstance(dustbin_bce_loss, torch.Tensor) else dustbin_bce_loss
            )
            dustbin_k_mae_value = dustbin_k_mae.item() if isinstance(dustbin_k_mae, torch.Tensor) else dustbin_k_mae
            dustbin_k_balance_value = (
                dustbin_k_balance_err.item()
                if isinstance(dustbin_k_balance_err, torch.Tensor)
                else dustbin_k_balance_err
            )
            stage1_ss_base_aux_loss_value = (
                stage1_ss_base_aux_loss.item()
                if isinstance(stage1_ss_base_aux_loss, torch.Tensor)
                else float(stage1_ss_base_aux_loss)
            )
            stage1_ss_base_aux_weighted_value = (
                float(stage1_ss_base_aux_weight) * stage1_ss_base_aux_loss_value
                if stage == 1
                else 0.0
            )
            total_loss_value = total_loss.item()
            total_loss.backward()

            if stage == 1 and not stage1_grad_diag_printed:
                try:
                    core_model = _unwrap_model(model)
                    matcher_sq = 0.0
                    backbone_node_sq = 0.0
                    backbone_edge_sq = 0.0
                    matcher_has_grad = False
                    backbone_node_has_grad = False
                    backbone_edge_has_grad = False

                    for name, param in core_model.named_parameters():
                        if not param.requires_grad or param.grad is None:
                            continue
                        grad_det = param.grad.detach()
                        if not torch.isfinite(grad_det).all():
                            grad_det = torch.nan_to_num(grad_det, nan=0.0, posinf=0.0, neginf=0.0)
                        sq = float(torch.sum(grad_det * grad_det).item())
                        if name.startswith("node_layers."):
                            backbone_node_sq += sq
                            backbone_node_has_grad = True
                        elif name.startswith(("edge_layers.", "final_layers.")):
                            backbone_edge_sq += sq
                            backbone_edge_has_grad = True
                        elif not name.startswith(("encoder_k.", "final_row.", "final_col.")) and name != "bin_score":
                            matcher_sq += sq
                            matcher_has_grad = True

                    diag_msg = (
                        "[Stage1 GradDiag] "
                        f"ds_mat_topk.requires_grad={bool(outputs.get('ds_mat_topk', None) is not None and outputs['ds_mat_topk'].requires_grad)} "
                        f"ds_mat_matcher.requires_grad={bool(outputs.get('ds_mat_matcher', None) is not None and outputs['ds_mat_matcher'].requires_grad)} "
                        f"grad_norm_matcher={matcher_sq ** 0.5:.6e} "
                        f"grad_norm_backbone_node={backbone_node_sq ** 0.5:.6e} "
                        f"grad_norm_backbone_edge={backbone_edge_sq ** 0.5:.6e} "
                        f"has_grad_matcher={matcher_has_grad} "
                        f"has_grad_backbone_node={backbone_node_has_grad} "
                        f"has_grad_backbone_edge={backbone_edge_has_grad}"
                    )
                    print(diag_msg)
                    logger.info(diag_msg)
                except Exception as exc:
                    warn_msg = f"[Stage1 GradDiag] failed: {exc}"
                    print(warn_msg)
                    logger.warning(warn_msg)
                stage1_grad_diag_printed = True

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            optimizer.step()
            if optimizer_k is not None:
                optimizer_k.step()

            epoch_loss_sum += loss_value
            epoch_total_loss_sum += total_loss_value
            running_ks_loss += ks_loss_value
            running_dustbin_loss += dustbin_loss_value
            running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error
            running_dustbin_k_mse += dustbin_k_mse_value
            running_dustbin_bce += dustbin_bce_value
            running_dustbin_k_mae += dustbin_k_mae_value
            running_dustbin_k_balance += dustbin_k_balance_value
            running_stage1_ss_base_aux += stage1_ss_base_aux_loss_value if stage == 1 else 0.0
            running_stage1_ss_base_aux_weighted += stage1_ss_base_aux_weighted_value if stage == 1 else 0.0

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
            class0_prec = batch_scalars["class0_precision"]
            class1_prec = batch_scalars["class1_precision"]
            class0_rec = batch_scalars["class0_recall"]
            class1_rec = batch_scalars["class1_recall"]
            class0_f1 = batch_scalars["class0_f1"]
            class1_f1 = batch_scalars["class1_f1"]

            global_step = (epoch - start_epoch) * steps_per_epoch + batch_idx
            if iter_num % 5 == 0:
                writer.add_scalar('Train/Loss_Batch', loss_value, global_step)
                if stage == 1:
                    writer.add_scalar('Train/Stage1_SSBase_Aux_Loss_Batch', stage1_ss_base_aux_loss_value, global_step)
                    writer.add_scalar('Train/Stage1_SSBase_Aux_Weighted_Batch', stage1_ss_base_aux_weighted_value, global_step)
                writer.add_scalar('Train/KS_Loss_Batch', ks_loss_value, global_step)
                writer.add_scalar('Train/Dustbin_Loss_Batch', dustbin_loss_value, global_step)
                writer.add_scalar('Train/Dustbin_K_MSE_Loss_Batch', dustbin_k_mse_value, global_step)
                writer.add_scalar('Train/Dustbin_BCE_Loss_Batch', dustbin_bce_value, global_step)
                writer.add_scalar('Train/Dustbin_K_MAE_Batch', dustbin_k_mae_value, global_step)
                writer.add_scalar('Train/Dustbin_K_BalanceErr_Batch', dustbin_k_balance_value, global_step)
                writer.add_scalar('Train/Total_Loss_Batch', total_loss_value, global_step)
                writer.add_scalar('Train/Accuracy_Batch', acc, global_step)
                writer.add_scalar('Train/Precision_Batch', prec, global_step)
                writer.add_scalar('Train/Recall_Batch', rec, global_step)
                writer.add_scalar('Train/F1_Batch', f1, global_step)
                writer.add_scalar('Train/F1_Micro_Batch', micro_f1, global_step)
                writer.add_scalar('Train/F1_Macro_Batch', macro_f1, global_step)
                writer.add_scalar('Train/Class0_Precision_Batch', class0_prec, global_step)
                writer.add_scalar('Train/Class0_Recall_Batch', class0_rec, global_step)
                writer.add_scalar('Train/Class0_F1_Batch', class0_f1, global_step)
                writer.add_scalar('Train/Class1_Precision_Batch', class1_prec, global_step)
                writer.add_scalar('Train/Class1_Recall_Batch', class1_rec, global_step)
                writer.add_scalar('Train/Class1_F1_Batch', class1_f1, global_step)
                if batch_k_debug:
                    for key, value in batch_k_debug.items():
                        if key == "sample_count":
                            continue
                        writer.add_scalar(f'Train/KDebug/{key}', float(value), global_step)

                avg_loss = epoch_loss_sum / iter_num
                avg_ks_loss = running_ks_loss / iter_num
                avg_dustbin_loss = running_dustbin_loss / iter_num
                avg_dustbin_k_mse = running_dustbin_k_mse / iter_num
                avg_dustbin_bce = running_dustbin_bce / iter_num
                avg_dustbin_k_mae = running_dustbin_k_mae / iter_num
                avg_stage1_ss_base_aux = running_stage1_ss_base_aux / iter_num
                avg_stage1_ss_base_aux_weighted = running_stage1_ss_base_aux_weighted / iter_num
                avg_total_loss = epoch_total_loss_sum / iter_num

                if "ks_loss" in outputs and optimizer_k is not None:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                            f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                            f"dustbin_loss: {avg_dustbin_loss:.4f}, "
                            f"dustbin_k_mse: {avg_dustbin_k_mse:.4f}, "
                            f"dustbin_bce: {avg_dustbin_bce:.4f}, "
                            f"dustbin_k_mae: {avg_dustbin_k_mae:.4f}, "
                            f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}, "
                            f"P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}, "
                            f"MiF1: {micro_f1:.4f}, MaF1: {macro_f1:.4f}")
                else:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                            f"Loss: {avg_loss:.4f}, dustbin_loss: {avg_dustbin_loss:.4f}, "
                            f"dustbin_k_mse: {avg_dustbin_k_mse:.4f}, dustbin_bce: {avg_dustbin_bce:.4f}, "
                            f"Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}, "
                            f"MiF1: {micro_f1:.4f}, MaF1: {macro_f1:.4f}")
                if stage == 1:
                    log_msg += (
                        f", stage1_ss_base_aux: {avg_stage1_ss_base_aux:.4f}"
                        f", stage1_ss_base_aux_w: {avg_stage1_ss_base_aux_weighted:.4f}"
                    )
                print(log_msg)
                logger.info(log_msg)

    avg_epoch_loss = epoch_loss_sum / iter_num
    avg_ks_loss = running_ks_loss / iter_num
    avg_dustbin_loss = running_dustbin_loss / iter_num
    avg_dustbin_k_mse = running_dustbin_k_mse / iter_num
    avg_dustbin_bce = running_dustbin_bce / iter_num
    avg_dustbin_k_mae = running_dustbin_k_mae / iter_num
    avg_dustbin_k_balance = running_dustbin_k_balance / iter_num
    avg_stage1_ss_base_aux = running_stage1_ss_base_aux / iter_num
    avg_stage1_ss_base_aux_weighted = running_stage1_ss_base_aux_weighted / iter_num
    avg_total_loss = epoch_total_loss_sum / iter_num
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
    avg_class0_precision = float(epoch_metric_summary["per_class_precision"][0].item())
    avg_class1_precision = float(epoch_metric_summary["per_class_precision"][1].item())
    avg_class0_recall = float(epoch_metric_summary["per_class_recall"][0].item())
    avg_class1_recall = float(epoch_metric_summary["per_class_recall"][1].item())
    avg_class0_f1 = float(epoch_metric_summary["per_class_f1"][0].item())
    avg_class1_f1 = float(epoch_metric_summary["per_class_f1"][1].item())


    writer.add_scalar('Train/Loss_Epoch', avg_epoch_loss, epoch)
    writer.add_scalar('Train/KS_Loss_Epoch', avg_ks_loss, epoch)
    writer.add_scalar('Train/Dustbin_Loss_Epoch', avg_dustbin_loss, epoch)
    writer.add_scalar('Train/Dustbin_K_MSE_Loss_Epoch', avg_dustbin_k_mse, epoch)
    writer.add_scalar('Train/Dustbin_BCE_Loss_Epoch', avg_dustbin_bce, epoch)
    writer.add_scalar('Train/Dustbin_K_MAE_Epoch', avg_dustbin_k_mae, epoch)
    writer.add_scalar('Train/Dustbin_K_BalanceErr_Epoch', avg_dustbin_k_balance, epoch)
    if stage == 1:
        writer.add_scalar('Train/Stage1_SSBase_Aux_Loss_Epoch', avg_stage1_ss_base_aux, epoch)
        writer.add_scalar('Train/Stage1_SSBase_Aux_Weighted_Epoch', avg_stage1_ss_base_aux_weighted, epoch)
        writer.add_scalar('Train/Stage1_SSBase_Aux_Weight', float(stage1_ss_base_aux_weight), epoch)
    writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
    writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)
    writer.add_scalar('Train/Precision_Epoch', avg_precision, epoch)
    writer.add_scalar('Train/Recall_Epoch', avg_recall, epoch)
    writer.add_scalar('Train/F1_Epoch', avg_f1, epoch)
    writer.add_scalar('Train/F1_Micro_Epoch', avg_micro_f1, epoch)
    writer.add_scalar('Train/F1_Macro_Epoch', avg_macro_f1, epoch)
    writer.add_scalar('Train/Class0_Precision_Epoch', avg_class0_precision, epoch)
    writer.add_scalar('Train/Class0_Recall_Epoch', avg_class0_recall, epoch)
    writer.add_scalar('Train/Class0_F1_Epoch', avg_class0_f1, epoch)
    writer.add_scalar('Train/Class1_Precision_Epoch', avg_class1_precision, epoch)
    writer.add_scalar('Train/Class1_Recall_Epoch', avg_class1_recall, epoch)
    writer.add_scalar('Train/Class1_F1_Epoch', avg_class1_f1, epoch)


    log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
            f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Dustbin Loss: {avg_dustbin_loss:.4f}, "
            f"Avg Dustbin K-MSE: {avg_dustbin_k_mse:.4f}, Avg Dustbin BCE: {avg_dustbin_bce:.4f}, "
            f"Avg Dustbin K-MAE: {avg_dustbin_k_mae:.4f}, "
            f"Avg Total Loss: {avg_total_loss:.4f}, Acc: {avg_accuracy:.4f}, "
            f"P: {avg_precision:.4f}, R: {avg_recall:.4f}, F1: {avg_f1:.4f}, "
            f"MiF1: {avg_micro_f1:.4f}, MaF1: {avg_macro_f1:.4f}")
    if stage == 1:
        log_msg += (
            f", Stage1 SSBase Aux: {avg_stage1_ss_base_aux:.4f}"
            f", Stage1 SSBase Aux Weighted: {avg_stage1_ss_base_aux_weighted:.4f}"
        )
    print(log_msg)
    logger.info(log_msg)

    if k_debug_weight_sum > 0:
        avg_k_debug = {
            key: value / k_debug_weight_sum
            for key, value in k_debug_sums.items()
        }
        for key, value in avg_k_debug.items():
            writer.add_scalar(f'Train/KDebug_Epoch/{key}', float(value), epoch)
        k_epoch_parts = []
        for key in (
            "gt_k_mean",
            "pred_k_mean",
            "pred_k_round_mean",
            "pred_k_mae",
            "dustbin_k_mean",
            "dustbin_k_round_mean",
            "dustbin_k_pred_mae",
            "dustbin_k_mse_loss",
            "dustbin_bce_loss",
            "dustbin_k_balance_err",
        ):
            if key in avg_k_debug:
                k_epoch_parts.append(f"{key}={avg_k_debug[key]:.4f}")
        if k_epoch_parts:
            k_log_msg = f"Train K Debug Epoch {epoch}: " + ", ".join(k_epoch_parts)
            print(k_log_msg)
            logger.info(k_log_msg)

    # if k_debug_weight_sum > 0:
    #     avg_k_debug = {
    #         key: value / k_debug_weight_sum
    #         for key, value in k_debug_sums.items()
    #     }
    #     k_epoch_parts = []
    #     if "gt_k_mean" in avg_k_debug:
    #         k_epoch_parts.append(f"gtK={avg_k_debug['gt_k_mean']:.2f}")
    #     if "pred_k_mean" in avg_k_debug:
    #         k_epoch_parts.append(f"predK={avg_k_debug['pred_k_mean']:.2f}")
    #     if "pred_k_round_mean" in avg_k_debug:
    #         k_epoch_parts.append(f"predK(round)={avg_k_debug['pred_k_round_mean']:.2f}")
    #     if "pred_zero_k_rate" in avg_k_debug:
    #         k_epoch_parts.append(f"predK0={avg_k_debug['pred_zero_k_rate']:.3f}")
    #     if "selected_zero_k_rate" in avg_k_debug:
    #         k_epoch_parts.append(f"selK0={avg_k_debug['selected_zero_k_rate']:.3f}")
    #     if "pred_k_mae" in avg_k_debug:
    #         k_epoch_parts.append(f"predK_MAE={avg_k_debug['pred_k_mae']:.2f}")
    #     if k_epoch_parts:
    #         k_log_msg = f"Train K Debug Epoch {epoch}: " + ", ".join(k_epoch_parts)
    #         print(k_log_msg)
    #         logger.info(k_log_msg)

    # Save model and optimizer states
    save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))


    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
