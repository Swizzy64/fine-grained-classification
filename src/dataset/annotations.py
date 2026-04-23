from scipy.io import loadmat
import os

def load_annotations(mat_file, img_dir):
    """
    Load annotations from a .mat file.

    Args:
        mat_file (str): Path to the .mat file containing annotations.
        img_dir (str): Path to the directory containing the images.

    Returns:
        dict: A dictionary containing the loaded annotations.
    """
    data = loadmat(mat_file)
    annotations = data["annotations"][0]
        
    samples = []
    for ann in annotations:
        img_name = ann["fname"][0]
        label = int(ann["class"][0][0]) - 1  # Convert to 0-based index

        img_directory = os.path.join(img_dir, img_name)
        samples.append((img_directory, label))
    
    return samples