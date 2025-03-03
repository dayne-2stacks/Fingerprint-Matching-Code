from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
import torch
from src.loss_func import PermutationLoss
import torch.optim as optim
from utils.models_sl import save_model, load_model
from utils.visualize import visualize_stochastic_matrix

from pathlib import Path
import cv2
import numpy as np


dataset_len = 640

benchmark = L3SFV2AugmentedBenchmark(
        sets='train',
        obj_resize=(320, 240),
        train_root='/green/data/L3SF_V2/L3SF_V2_Augmented'
    )

print(benchmark.get_path("R1_47_right_loop_aug_1"))

image_dataset = GMDataset("L3SFV2Augmented",
                     benchmark,
                     dataset_len,
                     True,
                     None,
                     "2GM")

dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=True)

model = Net()

# Initialize loss function
criterion = PermutationLoss()


LR = 2.e-4
BACKBONE_LR = 2.e-5
K_LR=1.e-4



model_params = model.parameters()

backbone_ids = [id(item) for item in model.backbone_params]
k_params = model.k_params_id
other_params = [param for param in model.parameters() if id(param) not in k_params and id(param) not in backbone_ids]

model_params = [
    {'params': other_params},
    {'params': model.backbone_params, 'lr': BACKBONE_LR}
]
k_reg_params = model.k_params
optimizer_k = optim.Adam(k_reg_params, lr=K_LR)


optimizer = optim.Adam(model_params, lr=LR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





model.to(device)
# model = DataParallel(model, device_ids=[0,1])

num_epochs = 40
start_epoch = 0

checkpoint_path = Path("result") / 'params'
checkpoint_path.mkdir(parents=True, exist_ok=True)

model_path  = ''
optim_path  = ''
optim_k_path = ''

if start_epoch != 0:
    model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
    optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if optimizer_k is not None:
        optim_k_path = str(checkpoint_path / 'optim_k_{:04}.pt'.format(start_epoch))
        # optim_k_path=str(checkpoint_path / "optim_k.pt")
        

if len(model_path) > 0:
    print('Loading model parameters from {}'.format(model_path))
    load_model(model, model_path, strict=False)
if len(optim_path) > 0:
    print('Loading optimizer state from {}'.format(optim_path))
    optimizer.load_state_dict(torch.load(optim_path))
    
if len(optim_k_path) > 0:
        try:
            print('Loading optimizer_k state from {}'.format(optim_k_path))
            optimizer_k.load_state_dict(torch.load(optim_k_path))
        except FileNotFoundError:
            print('Creating new optimizer for AFA modules')

if optimizer_k is not None:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[10, 20, 30],
                                            gamma=0.1,
                                            last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1
    scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k,
                                            milestones=[10, 20, 30],
                                            gamma=0.1,
                                            last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1




### EARLY STOPPING LOGIC ###
patience = 3                 # Number of epochs to wait for improvement
best_loss = float('inf')     # Track the best (lowest) loss so far
no_improvement_count = 0     # How many epochs in a row we haven't improved

# Because your code uses one sample repeatedly, we’ll fetch that once:
single_sample = next(iter(dataloader))
single_sample = data_to_cuda(single_sample)  # Move sample to GPU if available


print(single_sample["id_list"])
print(single_sample.keys())

print("Kpts: ", single_sample["ns"])



for epoch in range(start_epoch, num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train()
    
    print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))
    if optimizer_k is not None:
        print('K_regression_lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer_k.param_groups]))
    # Track cumulative loss within this epoch
    epoch_loss_sum = 0.0
    running_loss = 0.0
    running_ks_loss = 0.0
    running_ks_error = 0
    
    num_iterations = 100  # fixed number of iterations you used
    
    for iter_num in range(num_iterations):
        optimizer.zero_grad()
        if optimizer_k is not None:
            optimizer_k.zero_grad()
            
        with torch.set_grad_enabled(True):

            outputs = model(single_sample)
        
            # Compute the loss
            loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
            
            if 'ks_loss' in outputs:
                    ks_loss = outputs['ks_loss']
                    ks_error = outputs['ks_error']
            # compute accuracy

        loss.backward()
        if optimizer_k is not None:
            ks_loss.backward()
        
        optimizer.step()
        if optimizer_k is not None:
            optimizer_k.step()
        
        epoch_loss_sum += loss.item()
        if 'ks_loss' in outputs:
            running_ks_loss += ks_loss
            running_ks_error += ks_error
        
        if iter_num % 5 == 0:
            print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(
                epoch, iter_num, loss.item()))
    
    # Average epoch loss
    epoch_loss = epoch_loss_sum / num_iterations
    print(f"==> End of Epoch {epoch}, average loss = {epoch_loss:.4f}")

    # Save model
    save_model(model, str(checkpoint_path / f'params_{epoch + 1:04}.pt'))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f'optim_{epoch + 1:04}.pt'))
    if optimizer_k is not None:
        torch.save(optimizer_k.state_dict(), str(checkpoint_path / f'optim_k_{epoch + 1:04}.pt'))
    
    # LR Scheduler step
    scheduler.step()
    if optimizer_k is not None:
        scheduler_k.step()

    ### EARLY STOPPING LOGIC ###
    if epoch_loss < best_loss:
        # Found improvement
        best_loss = epoch_loss
        no_improvement_count = 0
        
        # Save best model separately
        best_model_path = str(checkpoint_path / 'best_model.pt')
        save_model(model, best_model_path)
    else:
        # No improvement this epoch
        no_improvement_count += 1
        print(f"No improvement for {no_improvement_count} epoch(s). Best loss so far: {best_loss:.4f}")
        
        # If we've gone 'patience' epochs without improvement, stop training
        if no_improvement_count >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement for {patience} epochs.")
            break
        

best_model_path = str(checkpoint_path / 'params_0019.pt')
print("Loading the best model for evaluation...")
load_model(model, best_model_path, strict=False)
# Final step: Evaluate the model on a sample and perform matching on the original images
model.eval()
with torch.no_grad():
    outputs = model(single_sample)
    
# Assume outputs['ds_mat'] is a similarity matrix between keypoints of image0 and image1.
# Also assume that your sample contains keypoint coordinates (e.g. "kp0" and "kp1")
# For demonstration, if these keys do not exist, we provide dummy keypoints.
if 'Ps' in single_sample:
    # Retrieve keypoints from the sample and move them to CPU/numpy
    kp0 = single_sample['Ps'][0].cpu().numpy().squeeze(0)
    kp1 = single_sample['Ps'][1].cpu().numpy().squeeze(0)
    print("Ps in sample")
else:
    # Replace these dummy coordinates with your actual keypoints if not stored in sample.
    kp0 = np.array([[100, 100], [150, 150], [200, 200]])
    kp1 = np.array([[110, 110], [160, 160], [210, 210]])



# Convert the model's outputs to numpy arrays
ds_mat = outputs['ds_mat'].cpu().numpy().squeeze(0)  # affinity scores
per_mat = outputs['perm_mat'].cpu().numpy().squeeze(0)  # binary valid match indicators

# Debug prints to verify dimensions
print("ds_mat shape:", ds_mat.shape)
print(ds_mat)
print("per_mat shape:", per_mat.shape)
print(per_mat)
visualize_stochastic_matrix(per_mat, "Perm_matrix")
print("Number of keypoints in image0 (from kp0):", kp0.shape[0])
print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

# Extract valid matches using per_mat as a mask
matches = []
for i in range(ds_mat.shape[0]):
    # Get indices in image1 that are flagged as a valid match for keypoint i
    valid_indices = np.where(per_mat[i] == 1)[0]
    if valid_indices.size == 0:
        # If no valid match is flagged, you can skip this keypoint
        continue
    # If multiple valid matches exist, select the one with the highest affinity score
    best_index = valid_indices[np.argmax(ds_mat[i, valid_indices])]
    distance_value = np.squeeze(ds_mat[i, best_index])
    if hasattr(distance_value, 'size') and distance_value.size != 1:
        distance_value = distance_value.flatten()[0]
    distance = float(distance_value)
    matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_index, _imgIdx=0, _distance=distance))

# Now load the original images (either from single_sample or use fallback paths)
if 'img0_path' in single_sample and 'img1_path' in single_sample:
    img0 = cv2.imread(single_sample['img0_path'])
    img1 = cv2.imread(single_sample['img1_path'])
else:
    img0 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg')
    img1 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg')
    print("Using fallback image paths.")


cv2_kp0 = []
for kp in kp0:
    x = float(np.array(kp[0]).flatten()[0])
    y = float(np.array(kp[1]).flatten()[0])
    cv2_kp0.append(cv2.KeyPoint(x=x, y=y, size=1))

cv2_kp1 = []
for kp in kp1:
    x = float(np.array(kp[0]).flatten()[0])
    y = float(np.array(kp[1]).flatten()[0])
    cv2_kp1.append(cv2.KeyPoint(x=x, y=y, size=1))
    
# Draw keypoints on both images
img0_kp = cv2.drawKeypoints(img0, cv2_kp0, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kp = cv2.drawKeypoints(img1, cv2_kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save or display the images
cv2.imwrite("image0_keypoints.jpg", img0_kp)
cv2.imwrite("image1_keypoints.jpg", img1_kp)

# Optional: print the number of keypoints to confirm they match expected dimensions.
print("cv2_kp0 length:", len(cv2_kp0))
print("cv2_kp1 length:", len(cv2_kp1))
print("Number of matches found:", len(matches))

# Draw the matches. The flag "flags=2" tells OpenCV to draw only the matching lines.
img_matches = cv2.drawMatches(img0, cv2_kp0, img1, cv2_kp1, matches, None, flags=2)

# Save the resulting image to disk.
cv2.imwrite("matching_result.jpg", img_matches)
print("Matching result saved as 'matching_result.jpg'.")