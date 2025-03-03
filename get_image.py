import cv2
import json
import numpy as np

# Open and read the JSON file
with open('/gold/home/dayneguy/Fingerprint/data/L3SFV2Augmented/train-(320, 240).json', 'r') as file:
    data = json.load(file)

# Print the data
kp0 = data['R1_8_right_loop_aug_0']["kpts"]
kp1 = data['R1_8_right_loop_aug_1']["kpts"]
print(kp0)

img0 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg')
img1 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg')

cv2_kp0 = []
for kp in kp0:
    print(kp)
    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    cv2_kp0.append(cv2.KeyPoint(x=x, y=y, size=1))

cv2_kp1 = []
for kp in kp1:
    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    cv2_kp1.append(cv2.KeyPoint(x=x, y=y, size=1))
    
# Draw keypoints on both images
img0_kp = cv2.drawKeypoints(img0, cv2_kp0, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kp = cv2.drawKeypoints(img1, cv2_kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2_kp0 = []
filtered_kp0_indices = []  # To track valid indices for matching

for i, kp in enumerate(kp0):
    if "labels" in kp and kp["labels"] == "outlier":
        continue  # Skip outliers

    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    cv2_kp0.append(cv2.KeyPoint(x=x, y=y, size=1))
    filtered_kp0_indices.append(i)  # Keep track of non-outlier indices

cv2_kp1 = []
filtered_kp1_indices = []

for i, kp in enumerate(kp1):
    if "labels" in kp and kp["labels"] == "outlier":
        continue  

    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    cv2_kp1.append(cv2.KeyPoint(x=x, y=y, size=1))
    filtered_kp1_indices.append(i)  

cv2_kp0 = []
filtered_kp0_indices = {}  


for i, kp in enumerate(kp0):
    if "labels" in kp and kp["labels"] == "outlier":
        continue 

    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    pore_id = kp["labels"]  

    cv2_kp0.append(cv2.KeyPoint(x=x, y=y, size=1))
    filtered_kp0_indices[pore_id] = len(cv2_kp0) - 1 

cv2_kp1 = []
filtered_kp1_indices = {}  

for i, kp in enumerate(kp1):
    if "labels" in kp and kp["labels"] == "outlier":
        continue  # Skip outliers

    x = float(np.array(kp["x"]).flatten()[0])
    y = float(np.array(kp["y"]).flatten()[0])
    pore_id = kp["labels"]  # Extract the unique pore ID

    cv2_kp1.append(cv2.KeyPoint(x=x, y=y, size=1))
    filtered_kp1_indices[pore_id] = len(cv2_kp1) - 1  # Store index using pore ID

# Create matches based on ground truth pore IDs
matches = []
for pore_id, idx0 in filtered_kp0_indices.items():
    if pore_id in filtered_kp1_indices:  # Check if the same ID exists in image 1
        idx1 = filtered_kp1_indices[pore_id]
        distance_value = 0.0  # Since it's a ground truth match, we assume perfect alignment

        match = cv2.DMatch(_queryIdx=idx0, _trainIdx=idx1, _imgIdx=0, _distance=distance_value)
        matches.append(match)



# Draw matches
img_matches = cv2.drawMatches(img0, cv2_kp0, img1, cv2_kp1, matches, None, flags=2)
cv2.imwrite("filtered_matching_result.jpg", img_matches)


print(len(filtered_kp1_indices))
# Save or display the images
cv2.imwrite("image0_keypoints.jpg", img0_kp)
cv2.imwrite("image1_keypoints.jpg", img1_kp)

