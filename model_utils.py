from typing import List, Tuple, Dict, Optional
from glob import glob
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from PIL import Image
import torch

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Detection Model Label Config
model_config_id2label ={0: 'table', 1: 'table rotated'}


def plot_results(pil_img, scores, labels, boxes, *, show_axis=False, show_grid=True, figsize=(20,20), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model_config_id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis("on" if show_axis else "off")
    if show_grid:
        plt.grid()
    if title:
        plt.title(title)
    plt.show()

class TableDetector:
    """Custom Class for table detection, visualization, etc"""
    def __init__(self, model, feature_extractor, *, threshold=0.7):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold

    def read_image(self, image_path, *, resize:Optional[Tuple[int]]=None):
        image = Image.open(image_path).convert("RGB")
        if resize:
            rx, ry = resize
            width, height = image.size
            image.resize((int(width*rx), int(height*ry)))

        return image

    def get_detections(self, image_path, *, resize:Optional[Tuple[int]]=None, threshold:Optional[float]=None, **kwargs):
        image = self.read_image(image_path, resize=resize)

        encoding = self.feature_extractor(image, return_tensors="pt")
        encoding.keys()
        with torch.no_grad():
            outputs = self.model(**encoding)

        # rescale bounding boxes
        width, height = image.size
        
        if threshold is None:
            threshold = self.threshold
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[(height, width)])[0]
        
        return results


    def plot_detections(self, image_path, *, resize:Optional[Tuple[int]]=None, **kwargs):
        results = self.get_detections(image_path, resize=resize, **kwargs)

        image = self.read_image(image_path, resize=resize)
        plot_results(image, results['scores'], results['labels'], results['boxes'], **kwargs)

