import json
from collections import Counter
from networkx.utils import UnionFind

class Image_Keypoints:
    def __init__(self, image):
        self.image = image
        self.keypoints = {}
        self.counter = 0

    def get_keypoint(self, keypoint: str) -> str:
        kp_str = map_keypoints(keypoint)
        if self.keypoints.get(kp_str) is None:
            self.counter +=1
            self.keypoints[kp_str] = self.counter
        return f'{self.image}_{self.keypoints[kp_str]}'

    def __len__(self):
        return self.counter


def map_keypoints(kp: tuple) -> str:
    return ",".join(str(int(v)) for v in kp)



def subject_pore_labels( subject_dict):
    
    uf = UnionFind()
    
    global_label = {}
    kp_desc = {}
    kps = {}

    def get_keypoint_labels(image: str, img_idx: int, anno: dict, kp_desc: dict, kps: dict) -> Image_Keypoints:
        desc = kp_desc.get(image)
        if desc is None:
            kp_desc[image] = Image_Keypoints(image)
            desc = kp_desc[image]
            # get nodes in image 1
            if img_idx == 1:
                for kp in anno['img1_keypoints']:
                    desc.get_keypoint(kp)
                    if kps.get(image) is None:
                        kps[image] = [kp]
                    else:
                        kps[image].append(kp)
            if img_idx == 2:
                for kp in anno['img2_keypoints']:
                    desc.get_keypoint(kp)
                    if kps.get(image) is None:
                        kps[image] = [kp]
                    else:
                        kps[image].append(kp)
        return desc
        
    for subject_id in subject_dict.keys():
        annos = subject_dict[subject_id]
        for anno in annos:
            # build descriptor that has universal name for each image keypoint
            desc1 = get_keypoint_labels(anno['image1'], 1, anno, kp_desc, kps)
            desc2 = get_keypoint_labels(anno['image2'], 2, anno, kp_desc, kps)

            for match in anno['matches']:
                kp1 = match['img1_point_rc']
                kp2 = match['img2_point_rc']
                uf.union(
                        desc1.get_keypoint(kp1),
                        desc2.get_keypoint(kp2)
                        )
                
    for k in uf:
        global_label[k] = uf[k]

    return global_label, kp_desc, kps