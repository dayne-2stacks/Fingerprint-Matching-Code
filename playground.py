from feature_extractor import VGG16_bn_final as CNN
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
import random
from PIL import Image

from spline_conv import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from feature_align import feature_align

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

with Image.open("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/1_left_loop_aug_0.jpg") as obj:
    obj = obj.resize((224, 224), resample=Image.BICUBIC,
                                 box=(0, 0, 320, 240))
    img = np.array(obj)
    img = trans(img)
    
    

cnn = CNN()

print(img.dim())

img = img.unsqueeze(0)
nodes = cnn.node_layers(img)
print("Nodes: ",nodes)
edges = cnn.edge_layers(nodes)
print("edges: ",edges)
global_feat = cnn.final_layers(edges).reshape((nodes.shape[0], -1))

print("global_feat: ",global_feat)


# L2 Norm
nodes = normalize_over_channels(nodes)
edges = normalize_over_channels(edges)

# arrange features
U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
node_features = torch.cat((U, F), dim=1)
node_feature_list.append(node_features.detach())
graph.x = node_features

graph = SiameseSConvOnNodes(graph)
orig_graph = SiameseNodeFeaturesToEdgeFeatures(graph)

print(graph)
print(orig_graph)            