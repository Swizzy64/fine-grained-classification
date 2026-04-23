from scipy.io import loadmat
import os

def load_annotations(mat_file, img_dir):
    data = loadmat(mat_file)
    annotations = data["annotations"][0]
        
    samples = []
    for ann in annotations:
        img_name = ann["fname"][0]
        label = int(ann["class"][0][0]) - 1

        img_directory = os.path.join(img_dir, img_name)
        samples.append((img_directory, label))
    
    return samples