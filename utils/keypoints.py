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



def subject_pore_labels(subject_dict, max_sq_alignment_error: float = 25.0):
    
    uf = UnionFind()
    component_members = {}
    
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
            seen_img1 = set()
            seen_img2 = set()

            for match in sorted(anno['matches'], key=lambda m: (m.get('sq_alignment_error', float('inf')), m.get('descriptor_distance', float('inf')))):
                if match.get('sq_alignment_error', float('inf')) > max_sq_alignment_error:
                    break
                kp1 = match['img1_point_rc']
                kp2 = match['img2_point_rc']
                kp1_key = map_keypoints(kp1)
                kp2_key = map_keypoints(kp2)
                if kp1_key in seen_img1 or kp2_key in seen_img2:
                    continue
                seen_img1.add(kp1_key)
                seen_img2.add(kp2_key)
                node1 = desc1.get_keypoint(kp1)
                node2 = desc2.get_keypoint(kp2)
                root1 = uf[node1]
                root2 = uf[node2]

                members1 = component_members.setdefault(root1, {})
                members1.setdefault(desc1.image, node1)
                members2 = component_members.setdefault(root2, {})
                members2.setdefault(desc2.image, node2)

                if root1 == root2:
                    continue

                if any(
                    image in members1 and members1[image] != node
                    for image, node in members2.items()
                ):
                    continue

                uf.union(node1, node2)
                new_root = uf[node1]
                merged_members = dict(members1)
                merged_members.update(members2)
                component_members[new_root] = merged_members
                if root1 != new_root:
                    component_members.pop(root1, None)
                if root2 != new_root:
                    component_members.pop(root2, None)
                
    for k in uf:
        global_label[k] = uf[k]

    return global_label, kp_desc, kps
