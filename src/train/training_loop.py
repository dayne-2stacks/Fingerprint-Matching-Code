import torch
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import save_model
from src.evaluation_metric import matching_accuracy
from src.model.dustbin import strip_dustbin




def train_epoch(model, dataloader, criterion, optimizer, optimizer_k,
                device, writer, epoch, start_epoch, stage, logger, checkpoint_path,
):
    # Initialize running sums and counters
    epoch_loss_sum = 0.0
    running_ks_loss = 0.0
    running_ks_error = 0.0
    epoch_accuracy_sum = 0.0
    epoch_total_loss_sum = 0.0
    iter_num = 0

    torch.autograd.set_detect_anomaly(True)
    for i in range(3):
        for batch_idx, batch in enumerate(dataloader):
            iter_num += 1

            # Send data to device
            batch = data_to_cuda(batch)

            # Zero gradients
            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()
                
            # Forward pass
            outputs = model(batch)

            outputs["ds_mat"] = strip_dustbin(outputs["ds_mat"])
            outputs["perm_mat"] = strip_dustbin(outputs["perm_mat"])
            outputs["gt_perm_mat"] = strip_dustbin(outputs["gt_perm_mat"])
            outputs["ns"] = [n - 1 for n in outputs["ns"]]


            # compute loss and their gradients
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
            ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))
            total_loss = loss +(ks_loss if isinstance(ks_loss, torch.Tensor) else torch.tensor(ks_loss, device=device)) 
            
            loss_value = loss.item()
            ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
            total_loss_value = total_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Adjust Learning weights
            optimizer.step()
            if optimizer_k is not None:
                optimizer_k.step()

            # Update running sums
            epoch_loss_sum += loss_value
            epoch_total_loss_sum += total_loss_value
            running_ks_loss += ks_loss_value
            running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error

        
            # Report accuracy
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

                avg_loss = epoch_loss_sum / iter_num
                avg_ks_loss = running_ks_loss / iter_num
                avg_total_loss = epoch_total_loss_sum / iter_num

                if "ks_loss" in outputs and optimizer_k is not None:
                    log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                              f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f},  "
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
              f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}, ")
    print(log_msg)
    logger.info(log_msg)

    # Save model and optimizer states
    save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))


    return avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy
