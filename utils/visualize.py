import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import seaborn as sns
import torch
import cv2
import numpy as np 

def visualize_stochastic_matrix(matrix, filename="matrix"):
    """
    Visualizes a stochastic matrix using a heatmap.
    
    Args:
        matrix (torch.Tensor or np.ndarray): Stochastic matrix to visualize.
    """
    # Convert tensor to NumPy (handle GPU tensors)
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if matrix.ndim == 3:
        matrix = matrix[0]  # Take the first matrix in the batch

    # matrix= matrix[:50][:50][:50]
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", annot=False, linewidths=0.5, cbar=True)

    plt.xlabel("Columns (Probability Distribution)")
    plt.ylabel("Rows")
    plt.title("Stochastic Matrix Heatmap")
    
    
    save_name= f"photos/{filename}.png"
  

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f"Graph visualization saved as {save_name}")


# Ensure the 'photos' directory exists
os.makedirs("photos", exist_ok=True)

def visualize_pyg_data(data, filename=None, layout="spring"):
    """
    Visualize a PyTorch Geometric Data object (or a list of Data objects) as graphs and save them as PNG files.
    
    Parameters:
        data (torch_geometric.data.Data or list): A single PyG Data object or a list of them.
        filename (str): The base filename for the saved PNG image(s). If data is a list and filename is provided,
                        each graph will be saved as photos/<filename>_i.png where i is the index.
        layout (str): Which layout to use for positioning the nodes ('spring', 'kamada_kawai', etc.).
    """
    # If data is a list, iterate over each element.
    if isinstance(data, list):
        for i, d in enumerate(data):
            fname = f"photos/{filename}_{i}.png" if filename else f"photos/graph_{i}.png"
            visualize_pyg_data(d, filename=fname, layout=layout)
        return

    # data is assumed to be a single PyG Data object.
    G = to_networkx(data, to_undirected=True)
    
    # Choose a layout for node positioning.
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)
    
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=50,
        node_color='skyblue',
        edge_color='gray',
        alpha=0.7
    )
    plt.title("Graph Visualization")
    
    save_name = filename if filename.startswith("photos/") else f"photos/{filename}"

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f"Graph visualization saved as {save_name}")


def visualize_pyg_batch(data, filename="photos/batch_graph.png"):
    """
    Visualize a PyG Batch (or Data) object as a graph and save it as a PNG file.
    
    Parameters:
        data (torch_geometric.data.Data or Batch): A PyG data object containing at least
            'edge_index' and 'x'.
        filename (str): The filename for the saved PNG.
    """
    # Convert the PyG object to a NetworkX graph.
    G = to_networkx(data, to_undirected=True)
    
    # Create a figure.
    plt.figure(figsize=(12, 12))
    
    # Compute a layout for the nodes.
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph.
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=50,
        node_color='skyblue',
        edge_color='gray',
        alpha=0.7
    )
    
    # Save the figure as a PNG in the 'photos' directory.
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f"Graph visualization saved as {filename}")
    


def visualize_match(img0, img1, kp0, kp1, matches, prefix="", filename="matching_result"):
    """
    Visualizes keypoints and matches between two images and saves the results.
    img0, img1: images as numpy arrays (grayscale or color)
    kp0, kp1: keypoints as Nx2 numpy arrays
    matches: list of cv2.DMatch objects
    prefix: optional string to prepend to output filenames
    """
    # Convert keypoints to cv2.KeyPoint objects
    cv2_kp0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp0]
    cv2_kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp1]

    img0_kp = cv2.drawKeypoints(img0, cv2_kp0, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img1_kp = cv2.drawKeypoints(img1, cv2_kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(f"{prefix}image0_keypoints.jpg", img0_kp)
    cv2.imwrite(f"{prefix}image1_keypoints.jpg", img1_kp)

    print("cv2_kp0 length:", len(cv2_kp0))
    print("cv2_kp1 length:", len(cv2_kp1))
    print("Number of matches found:", len(matches))

    img_matches = cv2.drawMatches(img0, cv2_kp0, img1, cv2_kp1, matches, None, flags=2)
    cv2.imwrite(f"{prefix}{filename}.jpg", img_matches)
    print(f"Matching result saved as '{prefix}{filename}.jpg'.")
