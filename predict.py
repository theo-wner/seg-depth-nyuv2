import torch
from model import SegDepthFormer
from dataset import NYUv2Dataset
from tqdm import tqdm
import transformers

from plot_utils import visualize_img_label_depth
import config

"""
Predicts with the model
"""

if __name__ == '__main__':
    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the model (and load it to the CPU)
    model = SegDepthFormer()

    if config.CPU_USAGE:
        checkpoint = torch.load(config.CHECKPOINT, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = SegDepthFormer.load_from_checkpoint(config.CHECKPOINT)
        model = model.to(config.DEVICES[0])
    model.eval()

    # Dataset
    dataset = NYUv2Dataset(split='test')

    # Predict
    for i in tqdm(range(100)):
        image, label, depth = dataset[i]

        if not config.CPU_USAGE:
            image = image.to(config.DEVICES[0])

        with torch.no_grad():
            seg_logits, depth_preds = model(image.unsqueeze(0))
            
            seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
            depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)
            
            seg_preds = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze()
            depth_preds = depth_preds.squeeze()

        # Mask out the predictions where the original image is black in case you predict augmented train images
        seg_preds[image.sum(dim=0) == 0] = 255
        depth_preds[image.sum(dim=0) == 0] = 0

        # Visualization
        image = image.cpu()
        depth = depth.cpu()
        label = label.cpu()
        seg_preds = seg_preds.cpu()
        depth_preds = depth_preds.cpu()
        visualize_img_label_depth(image, label, seg_preds, depth, depth_preds, filename=f'test_{i}.png')

