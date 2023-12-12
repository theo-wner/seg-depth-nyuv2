import transformers
import torch
from model import SegFormer, DepthFormer, SegDepthFormer
import config
from test_utils import evaluate_inference_time, predict
from plot_utils import *

'''
Evaluates the inference time of a model or predicts with the model
'''
if __name__ == '__main__':

    # Define the relevant pictures
    images = [136, 0, 166, 305, 138]

    # Define Lists to store the results
    images_seg = []
    gt_seg = []
    preds_seg = []

    images_depth = []
    gt_depth = []
    preds_depth = []

    images_segdepth = []
    gt_segdepth_label = []
    gt_segdepth_depth = []
    preds_segdepth_label = []
    preds_segdepth_depth = []

    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # ------------------- Segmentation -------------------
    # Define the checkpoint
    checkpoint = ...

    # Initialize the model
    model = SegFormer().to(config.DEVICES[0])
    
    # Load the model
    checkpoint = torch.load(checkpoint, map_location=torch.device(config.DEVICES[0]))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Predict with the model
    model.eval()
    
    # Use Generator to predict
    for image, label, depth, seg_preds, depth_preds in predict(model, images, 'seg'):
        images_seg.append(image.cpu())
        gt_seg.append(label.cpu())
        preds_seg.append(seg_preds.cpu())

    # ------------------- Depth -------------------
    # Define the checkpoint
    checkpoint = ...

    # Initialize the model
    model = DepthFormer().to(config.DEVICES[0])

    # Load the model
    checkpoint = torch.load(checkpoint, map_location=torch.device(config.DEVICES[0]))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Predict with the model
    model.eval()

    # Use Generator to predict
    for image, label, depth, seg_preds, depth_preds in predict(model, images, 'depth'):
        images_depth.append(image.cpu())
        gt_depth.append(depth.cpu())
        preds_depth.append(depth_preds.cpu())

    # ------------------- Segmentation + Depth -------------------
    # Define the checkpoint
    checkpoint = ...

    # Initialize the model
    model = SegDepthFormer().to(config.DEVICES[0])

    # Load the model
    checkpoint = torch.load(checkpoint, map_location=torch.device(config.DEVICES[0]))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Predict with the model
    model.eval()

    # Use Generator to predict
    for image, label, depth, seg_preds, depth_preds in predict(model, images, 'segdepth'):
        images_segdepth.append(image.cpu())
        gt_segdepth_label.append(label.cpu())
        gt_segdepth_depth.append(depth.cpu())
        preds_segdepth_label.append(seg_preds.cpu())
        preds_segdepth_depth.append(depth_preds.cpu())

    # ------------------- Visualization -------------------
    for i in range(len(images)):
        visualize_gts_prs(images_segdepth[i], gt_segdepth_label[i], gt_segdepth_depth[i], preds_seg[i], preds_depth[i], preds_segdepth_label[i], preds_segdepth_depth[i], filename=f'./figures/discussion_{i}.png')