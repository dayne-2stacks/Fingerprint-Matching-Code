#!/usr/bin/env python
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
import torch.optim as optim
import json
import os
from torch.utils.tensorboard import SummaryWriter


from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
from src.loss_func import PermutationLoss
from utils.models_sl import save_model, load_model
from utils.visualize import visualize_stochastic_matrix, visualize_match
from src.evaluation_metric import matching_accuracy
from utils.scheduler import WarmupScheduler
# from apex import amp


NORM_MEANS= [0.485, 0.456, 0.406] 
NORM_STD=[0.229, 0.224, 0.225]

def to_grayscale_cv2_image(tensor, mean=NORM_MEANS, std=NORM_STD):
    """
    Converts a CHW torch tensor (normalized in [0,1] or by mean/std) 
    to a uint8 OpenCV grayscale image.
    """
    tensor = tensor.detach().cpu()

    # 1) Undo Normalize(mean,std) if provided
    if mean is not None and std is not None:
        # assume mean/std are sequences of length = channels
        m = torch.tensor(mean).view(-1, 1, 1)
        s = torch.tensor(std).view(-1, 1, 1)
        tensor = tensor * s + m

    # 2) CHW → HWC and scale to [0,255]
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # 3) RGB → Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

start_epoch = float('inf')
# config_files = [ "stage1.yml", "stage2.yml","stage3.yml"]
config_files = [ "stage3.yml"]
start_path = Path("checkpoint1")
start_path.mkdir(parents=True, exist_ok=True)
start_file = start_path / "checkpoint.json"

# Create TensorBoard log directory
log_dir = Path("logs/tensorboard")
log_dir.mkdir(parents=True, exist_ok=True)

for file in config_files:
    scheduler = scheduler_k = None
    print("Using config ", file)
    # ====================================================
    # Load Settings from YAML Configuration File
    # =====================================================
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]

    # Create a new writer for this training stage
    writer = SummaryWriter(log_dir=str(log_dir / file.split('.')[0]))
    
    # Hyperparameters from config
    if os.path.exists(start_file):
        with open(start_file, "r") as f:
            start_data = json.load(f)
            start_epoch = start_data.get("start_epoch", 0)  # Default to 0 if not found
            print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = train_config.get("start_epoch", 0)
    num_iterations = train_config.get("num_iterations", 25)
    # The K_Optimize flag from config will be overridden based on stage below.
    BATCH_SIZE     = train_config.get("BATCH_SIZE", 1)
    LR             = train_config.get("LR", 2e-3)
    BACKBONE_LR    = train_config.get("BACKBONE_LR", 2e-5)
    print("BACKBONE_LR =", BACKBONE_LR)
    K_LR           = train_config.get("K_LR", 2e-3)
    LR_DECAY       = train_config.get("LR_DECAY", 0.5)
    patience       = train_config.get("patience", 10)
    num_epochs     = train_config.get("num_epochs", 10)
    
    ngm_config = config.get("ngm", {})
    REGRESSION    = ngm_config.get("REGRESSION", True)
    
    print("Start epoch: ", start_epoch)
    # =====================================================
    # Hard-Coded and Derived Parameters
    # =====================================================
    dataset_len = 640

    best_loss = float('inf')
    no_improvement_count = 0

    # File paths
    train_root = 'dataset/Synthetic'
    OUTPUT_PATH = "result1"
    PRETRAINED_PATH = "" 

    # =====================================================
    # Setup Logging
    # =====================================================
    logging.basicConfig(
        filename='fp1.log', 
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)

    # =====================================================
    # Dataset and Dataloader
    # =====================================================
    benchmark = L3SFV2AugmentedBenchmark(
        sets='train',
        obj_resize=(320, 240),
        train_root=train_root
    )
    
    test_bm =  L3SFV2AugmentedBenchmark(
        sets='test',
        obj_resize=(320, 240),
        train_root=train_root
    )
    val_bm = L3SFV2AugmentedBenchmark(
        sets='val',
        obj_resize=(320, 240),
        train_root=train_root
    )

    image_dataset = GMDataset("L3SFV2Augmented", benchmark, dataset_len, True, None, "2GM")
    test_dataset = GMDataset("L3SFV2Augmented", test_bm, dataset_len, True, None, "2GM")
    val_dataset = GMDataset("L3SFV2Augmented", val_bm, dataset_len, True, None, "2GM")

    dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=False)
    test_dataloader = get_dataloader(test_dataset, shuffle=True, fix_seed=False)
    val_dataloader = get_dataloader(val_dataset, shuffle=True, fix_seed=False)
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(regression=REGRESSION)
    criterion = PermutationLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Uncomment below if using multiple GPUs:
    # model = DataParallel(model, device_ids=[0,1])
    # model, optimizer = amp.initialize(model, optimizer)
    # =====================================================
    # Freeze / Unfreeze Layers Based on Stage
    # =====================================================
    # Determine the current stage based on the config file name.
    if "stage1" in file:
        stage = 1
    elif "stage2" in file:
        stage = 2
    elif "stage3" in file:
        stage = 3
    else:
        stage = None  # Default if not matching; ideally should not happen.



    # =====================================================
    # Set up Optimizers (with optional separate k_optimizer)
    # =====================================================
    # For stage 1, we update only non-k_params; for stage 2, only k_params are trainable.
    # In stage 3, we use the original parameter grouping.
    backbone_ids = [id(item) for item in model.backbone_params]
    k_params = model.k_params_id
    other_params = [param for param in model.parameters() if id(param) not in k_params and id(param) not in backbone_ids]
    model_params = [
        {'params': other_params},
        {'params': model.backbone_params, 'lr': BACKBONE_LR}
        ]
    if stage == 1:
        
        print("Stage 1: Freezing all layers in k_params and training other parameters.")
        # Freeze k_params
        for param in model.k_params:
            for p in param["params"]:
                p.requires_grad = False
        # # Ensure all other parameters are trainable
       # In stage 1 we do not optimize the k_params
        K_Optimize = False

        # Only parameters with requires_grad == True will be optimized.
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = None
    elif stage == 2:
    
        print("Stage 2: Freezing all parameters except k_params (which are unfrozen).")
        for name, param in model.named_parameters():
            if id(param) not in model.k_params_id:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # In stage 2, we optimize only the k_params.
        K_Optimize = True
      
        # In stage 2, only k_params are trainable.
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    elif stage == 3:
        # Stage 3: All parameters are trainable. We separate backbone parameters for a different LR.s
        
        
        print("Stage 3: Unfreezing all layers for full fine-tuning.")
        # Unfreeze every parameter
        for name,  param in model.named_parameters():
            param.requires_grad = True
        K_Optimize = True


        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        optimizer_k = None

    # =====================================================
    # Schedulers for Both Optimizers
    # =====================================================
    # milestones = [int(0.6 * num_epochs), int(0.7 * num_epochs), int(0.9 * num_epochs)]
    # milestones = [int(0.025*num_epochs),int(0.05*num_epochs),int(0.2*num_epochs),int(0.4*num_epochs),int(0.6*num_epochs),int(0.8*num_epochs),int(0.1*num_epochs)]
    warmup_epochs = 10
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            #    milestones=milestones,
                                            #    gamma=LR_DECAY,
                                            #    last_epoch=-1)
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=LR_DECAY)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)
    if optimizer_k is not None:
        # main_scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k,
        #                                               milestones=milestones,
        #                                               gamma=LR_DECAY,
        #                                               last_epoch=-1)
        main_scheduler_k = optim.lr_scheduler.ReduceLROnPlateau(optimizer_k, patience=1, factor=LR_DECAY)
        scheduler_k = WarmupScheduler(optimizer_k, warmup_epochs=2, after_scheduler=main_scheduler_k)

    # =====================================================
    # Checkpoint Loading (if start_epoch > 0)
    # =====================================================
    checkpoint_path = Path(OUTPUT_PATH) / 'params'
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_path = ""
    optim_path = ""
    optim_k_path = ""
    if start_epoch != 0:
        model_path = str(checkpoint_path / f'params_{start_epoch:04}.pt')
        optim_path = str(checkpoint_path / f'optim_{start_epoch:04}.pt')
        if optimizer_k is not None:
            optim_k_path = str(checkpoint_path / f'optim_k_{start_epoch:04}.pt')

    if len(PRETRAINED_PATH) > 0:
        model_path = PRETRAINED_PATH

    if len(model_path) > 0:
        print("Loading model parameters from {}".format(model_path))
        load_model(model, model_path)
    if len(optim_path) > 0:
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    if len(optim_k_path) > 0:
        try:
            print("Loading optimizer_k state from {}".format(optim_k_path))
            optimizer_k.load_state_dict(torch.load(optim_k_path))
        except FileNotFoundError:
            print("No optimizer_k checkpoint found; starting fresh for k_params.")
    
    # Initialize warmup scheduler learning rates after loading optimizer state
    # if start_epoch == 0:  # Only for fresh training, not resuming
    print("Initializing warmup learning rates for first epoch...")
    # Set the initial warmup learning rates
    initial_lr = scheduler.get_initial_lr() if hasattr(scheduler, 'get_initial_lr') else LR / warmup_epochs
    initial_backbone_lr = BACKBONE_LR / warmup_epochs  # Apply warmup to backbone LR too
    
    for param_group in optimizer.param_groups:
        if param_group == optimizer.param_groups[-1]:  # Backbone group (assuming it's last)
            param_group['lr'] = initial_backbone_lr
        else:  # Other parameter groups
            param_group['lr'] = initial_lr
    
    if optimizer_k is not None and scheduler_k is not None:
        initial_k_lr = scheduler_k.get_initial_lr() if hasattr(scheduler_k, 'get_initial_lr') else K_LR / warmup_epochs
        for param_group in optimizer_k.param_groups:
            param_group['lr'] = initial_k_lr
            
    
    best_model_path = str(checkpoint_path / "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path} before training loop...")
        load_model(model, best_model_path)
                
    

    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(start_epoch, start_epoch + num_epochs):
        logger.info(f"Epoch {epoch}/{start_epoch + num_epochs - 1}")
        logger.info("-" * 50)
        print("Epoch {}/{}".format(epoch, start_epoch + num_epochs - 1))
        print("-" * 10)
        
        model.train()
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))
        
        # Log learning rates to TensorBoard
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)
            
        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)
        
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
                
                # Zero the parameter gradients 
                optimizer.zero_grad()
                if optimizer_k is not None:
                    optimizer_k.zero_grad()
                
                outputs = model(batch)
                
                
                # -------------- Check ds mat ---------------------
                # # Print a summary of the outputs to see if they vary across iterations
                # ds_mat = outputs["ds_mat"]
                # print(f"Iteration {batch_idx}: ds_mat mean = {ds_mat.mean().item():.4f}, std = {ds_mat.std().item():.4f}")
                # -------------- Check ds mat ---------------------
                

                loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
                ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
                ks_error = outputs.get("ks_error", torch.tensor(0.0, device=device))
                
                
                loss_value = loss.item()
                ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else ks_loss
                total_loss = loss + (ks_loss if isinstance(ks_loss, torch.Tensor) else ks_loss)
                total_loss_value = total_loss.item()
                
                # Accumulate all losses consistently
                epoch_loss_sum += loss_value
                epoch_total_loss_sum += total_loss_value
                running_ks_loss += ks_loss_value
                running_ks_error += ks_error.item() if isinstance(ks_error, torch.Tensor) else ks_error
                
                total_loss.backward()
                
                # Gradient clipping
                if stage == 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)    
                optimizer.step()
                if optimizer_k is not None:
                    optimizer_k.step()
                
                # Compute accuracy (if desired)
                acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
                print("Accuracy", acc)
                # Convert tensor to scalar value if needed
                if isinstance(acc, torch.Tensor):
                    if acc.numel() > 1:  # Check if tensor has multiple elements
                        acc = acc.mean().item()  # Take the mean before converting to scalar
                    else:
                        acc = acc.item()
                epoch_accuracy_sum += acc
                
                # Log every few iterations to TensorBoard
                global_step = (epoch - start_epoch) * len(dataloader) * 3 + i * len(dataloader) + batch_idx
                if iter_num % 5 == 0:
                    writer.add_scalar('Train/Loss_Batch', loss_value, global_step)
                    writer.add_scalar('Train/KS_Loss_Batch', ks_loss_value, global_step)
                    writer.add_scalar('Train/Total_Loss_Batch', total_loss_value, global_step)
                    writer.add_scalar('Train/Accuracy_Batch', acc, global_step)
                
                if iter_num % 5 == 0:
                    # Calculate running averages
                    avg_loss = epoch_loss_sum / iter_num
                    avg_ks_loss = running_ks_loss / iter_num
                    avg_total_loss = epoch_total_loss_sum / iter_num
                    
                    if "ks_loss" in outputs and optimizer_k is not None:
                        log_msg = (f"Epoch: {epoch}, Iter: {iter_num}, "
                                  f"Loss: {avg_loss:.4f}, ks_loss: {avg_ks_loss:.4f}, "
                                  f"total_loss: {avg_total_loss:.4f}, Acc: {acc:.4f}")
                        print(log_msg)
                        logger.info(log_msg)
                    else:
                        log_msg = f"Epoch: {epoch}, Iter: {iter_num}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}"
                        print(log_msg)
                        logger.info(log_msg)
            
        avg_epoch_loss = epoch_loss_sum / iter_num
        avg_ks_loss = running_ks_loss / iter_num
        avg_total_loss = epoch_total_loss_sum / iter_num
        # Also ensure avg_accuracy is a scalar
        avg_accuracy = epoch_accuracy_sum / iter_num
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Train/Loss_Epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Train/KS_Loss_Epoch', avg_ks_loss, epoch)
        writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
        writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)
        
        log_msg = (f"==> End of Epoch {epoch}, Avg Primary Loss: {avg_epoch_loss:.4f}, "
                  f"Avg KS Loss: {avg_ks_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}")
        print(log_msg)
        logger.info(log_msg)
        
        
        # Save checkpoints for model, optimizer, and optimizer_k
        save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
        torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
        if optimizer_k is not None:
            torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))
            
        # =====================================================
        # ---- Validation after each epoch ----
        # =====================================================
        model.eval()
        val_loss_sum = 0.0
        val_ks_sum = 0.0
        val_total_sum = 0.0 
        val_num = 0
        val_accuracy_sum = 0.0
        
        with torch.no_grad():
            for batch in val_dataloader:
                val_num += 1
                batch = data_to_cuda(batch)
                outputs = model(batch)
                loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
                ks_loss = outputs.get("ks_loss", torch.tensor(0.0, device=device))
                
                # Get scalar values
                loss_value = loss.item()
                ks_loss_value = ks_loss.item() if isinstance(ks_loss, torch.Tensor) else float(ks_loss)
                total_loss_value = loss_value + ks_loss_value
                
                # Compute accuracy
                acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
                # Convert tensor to scalar value if needed
                if isinstance(acc, torch.Tensor):
                    if acc.numel() > 1:  # Check if tensor has multiple elements
                        acc = acc.mean().item()  # Take the mean before converting to scalar
                    else:
                        acc = acc.item()
               
                val_accuracy_sum += acc
                
                # Accumulate all values
                val_loss_sum += loss_value
                val_ks_sum += ks_loss_value
                val_total_sum += total_loss_value
                
                if val_num % 5 == 0:
                    print(f"Validation batch {val_num} - Loss: {loss_value:.4f}, KS Loss: {ks_loss_value:.4f}, Total Loss: {total_loss_value:.4f}")

        # Calculate averages
        avg_val_loss = val_loss_sum / len(val_dataloader)
        avg_ks_loss = val_ks_sum / len(val_dataloader)
        avg_val_total = val_total_sum / len(val_dataloader)
        avg_val_accuracy = val_accuracy_sum / len(val_dataloader)
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/KS_Loss', avg_ks_loss, epoch)
        writer.add_scalar('Validation/Total_Loss', avg_val_total, epoch)
        writer.add_scalar('Validation/Accuracy', avg_val_accuracy, epoch)
        
        # Log validation results
        log_msg = f"Epoch {epoch} Validation: Primary Loss = {avg_val_loss:.4f}, KS Loss = {avg_ks_loss:.4f}, Total Loss = {avg_val_total:.4f}"
        print(log_msg)
        logger.info(log_msg)
        
        # Compute accuracy on the last batch of validation
        acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
        if isinstance(acc, torch.Tensor):
                    if acc.numel() > 1:  # Check if tensor has multiple elements
                        acc = acc.mean().item()  # Take the mean before converting to scalar
                    else:
                        acc = acc.item()
             
        print(f"Validation Accuracy: {acc:.4f}")
    
        
        # Save best model based on validation loss and update checkpoint file
        if avg_val_total < best_loss:
            best_loss = avg_val_total
            no_improvement_count = 0
            best_model_path = str(checkpoint_path / "best_model.pt")
            save_model(model, best_model_path)
            with open(start_file, "w") as f:
                json.dump({"start_epoch": epoch + 1}, f)
        else:
            no_improvement_count += 1
            print("No improvement for {} epoch(s). Best loss so far: {:.4f}".format(no_improvement_count, best_loss))
            if no_improvement_count >= patience:
                print("Stopping early at epoch {} due to no improvement.".format(epoch + 1))
                break

        

        # Initialize previous learning rates on first epoch
        if epoch == start_epoch:
            prev_lr = [group['lr'] for group in optimizer.param_groups]
            if optimizer_k is not None:
                prev_k_lr = [group['lr'] for group in optimizer_k.param_groups]

        # Step LR schedulers
        scheduler.step(avg_val_loss)
        if optimizer_k is not None:
            scheduler_k.step(avg_ks_loss)

        # Detect LR reduction for main optimizer
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        lr_reduced = any(clr < plr for clr, plr in zip(curr_lr, prev_lr))
        prev_lr = curr_lr  # Update previous for next iteration

        if lr_reduced:
            print("[LR REDUCED] Reloading best model weights from", checkpoint_path / "best_model.pt")
            best_model_path = str(checkpoint_path / "best_model.pt")
            load_model(model, best_model_path)
            
       
        
        # ---- Test Evaluation Periodically ----
        if epoch % 10 == 0:
            model.eval()
            test_loss_sum = 0.0
            test_accuracy_sum = 0.0
            last_batch = None
            last_outputs = None
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    batch = data_to_cuda(batch)
                    outputs = model(batch)
                    loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
                    if isinstance(acc, torch.Tensor):
                        if acc.numel() > 1:  # Check if tensor has multiple elements
                            acc = acc.mean().item()  # Take the mean before converting to scalar
                        else:
                            acc = acc.item()
               
                    
                    test_loss_sum += loss.item()
                    test_accuracy_sum += acc
                    
                    if batch_idx == 0:  # Save the first batch for visualization
                        last_batch = batch
                        last_outputs = outputs

            avg_test_loss = test_loss_sum / len(test_dataloader)
            avg_test_accuracy = test_accuracy_sum / len(test_dataloader)
            
            # Log test metrics to TensorBoard
            writer.add_scalar('Test/Loss', avg_test_loss, epoch)
            writer.add_scalar('Test/Accuracy', avg_test_accuracy, epoch)
            
            print("Epoch {}: Test Loss = {:.4f}, Test Accuracy = {:.4f}".format(
                epoch, avg_test_loss, avg_test_accuracy))

            # --- Visualization for the last batch ---
            if last_batch is not None and last_outputs is not None:
                # Extract keypoints
                if 'Ps' in last_batch:
                    kp0 = last_batch['Ps'][0][0].cpu().numpy()
                    kp1 = last_batch['Ps'][1][0].cpu().numpy()
                else:
                    kp0 = np.array([[100, 100], [150, 150], [200, 200]])
                    kp1 = np.array([[110, 110], [160, 160], [210, 210]])

                ds_mat = last_outputs["ds_mat"].cpu().numpy()[0]
                per_mat = last_outputs["perm_mat"].cpu().numpy()[0]

                matches = []
                for i in range(ds_mat.shape[0]):
                    valid_indices = np.where(per_mat[i] == 1)[0]
                    if valid_indices.size == 0:
                        continue
                    best_index = valid_indices[np.argmax(ds_mat[i, valid_indices])]
                    distance_value = np.squeeze(ds_mat[i, best_index])
                    if hasattr(distance_value, "size") and distance_value.size != 1:
                        distance_value = distance_value.flatten()[0]
                    distance = float(distance_value)
                    matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_index, _imgIdx=0, _distance=distance))

                # Create a visualization of matches and add to TensorBoard
                if "id_list" in last_batch:
                    img0 = last_batch["images"][0][0]
                    img1 = last_batch["images"][1][0]
                else:
                    img0 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg")
                    img1 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg")

                img0 = to_grayscale_cv2_image(img0)
                img1 = to_grayscale_cv2_image(img1)
                
                # Visualize and save match image, then add to TensorBoard
                match_path = f"photos/test_photos/match_{epoch}.jpg"
                visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/test_photos/", filename=f"match_{epoch}.jpg")

                if os.path.exists(match_path):
                    match_img = cv2.imread(match_path)
                    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                    writer.add_image(f'Test/Matches', match_img.transpose(2, 0, 1), epoch, dataformats='CHW')
    
    # Close the TensorBoard writer at the end of this training stage
    writer.close()

# =====================================================
# Load Best Model and Evaluate on a Sample
# =====================================================
single_sample = next(iter(val_dataloader))
single_sample = data_to_cuda(single_sample)
print(single_sample.keys())

# os.remove(start_file)
best_model_path = str(checkpoint_path / "best_model.pt")
print("Loading the best model for evaluation...")
load_model(model, best_model_path)
model.eval()
with torch.no_grad():
    outputs = model(single_sample)
    
acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
if isinstance(acc, torch.Tensor):
    if acc.numel() > 1:  # Check if tensor has multiple elements
        acc = acc.mean().item()  # Take the mean before converting to scalar
    else:
        acc = acc.item()

                
    
# Explicitly select the first sample from the batch
if 'Ps' in single_sample:
    kp0 = single_sample['Ps'][0][0].cpu().numpy()
    kp1 = single_sample['Ps'][1][0].cpu().numpy()
    print("Ps in sample")
else:
    kp0 = np.array([[100, 100], [150, 150], [200, 200]])
    kp1 = np.array([[110, 110], [160, 160], [210, 210]])

print("Number of keypoints in image0 (kp0):", len(kp0))
print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

# Ensure keypoints lists for OpenCV are correctly formed:
cv2_kp0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp0]
cv2_kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp1]

ds_mat = outputs["ds_mat"].cpu().numpy()[0]
per_mat = outputs["perm_mat"].cpu().numpy()[0]

# print("ds_mat shape:", ds_mat.shape)
# print(ds_mat)
# print("per_mat shape:", per_mat.shape)
# print(per_mat)
# visualize_stochastic_matrix(per_mat, "Perm_matrix")
# print("Number of keypoints in image0 (from kp0):", kp0.shape[0])
# print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

matches = []
for i in range(ds_mat.shape[0]):
    valid_indices = np.where(per_mat[i] == 1)[0]
    if valid_indices.size == 0:
        continue
    best_index = valid_indices[np.argmax(ds_mat[i, valid_indices])]
    distance_value = np.squeeze(ds_mat[i, best_index])
    if hasattr(distance_value, "size") and distance_value.size != 1:
        distance_value = distance_value.flatten()[0]
    distance = float(distance_value)
    matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_index, _imgIdx=0, _distance=distance))
    
print(len(single_sample["images"]))

if "id_list" in single_sample:
    img0 = single_sample["images"][0][0]
    img1 = single_sample["images"][1][0]
else:
    img0 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg")
    img1 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg")
    print("Using fallback image paths.")




img0= to_grayscale_cv2_image(img0)
img1 = to_grayscale_cv2_image(img1)

visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/")
print("Accuracy: ", acc)

# Add final visualizations to TensorBoard
if len(matches) > 0:
    match_path = f"photos/final_match-train.jpg"
    visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/", filename="final_match-train")
    visualize_stochastic_matrix(ds_mat, filename="matrix-train")

    if os.path.exists(match_path):
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        writer.add_image('Final/Matches', match_img.transpose(2, 0, 1), dataformats='CHW')

writer.add_scalar('Final/Accuracy', acc, 0)
writer.close()

print("Accuracy: ", acc)
