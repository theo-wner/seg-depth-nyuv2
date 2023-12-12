from statistics import mean, stdev
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset import NYUv2Dataset
import config


'''
Evaluates the inference time of a model
'''
def evaluate_inference_time(model, repetitions, task):

    dataset = NYUv2Dataset(split='test')
    image, _, _ = dataset[0]
    image = image.unsqueeze(0).to(config.DEVICES[0])

    # inference time measurement
    times = []
    with torch.no_grad():
        for repetition in tqdm(range(repetitions)):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            starter.record()
            
            if task == 'seg':
                logits = model(image)
                upsampled_logits = torch.nn.functional.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                preds = torch.softmax(upsampled_logits, dim=1)
                preds = torch.argmax(preds, dim=1)

            elif task == 'depth':
                preds = model(image)
                preds = F.relu(preds, inplace=True)
                preds = torch.nn.functional.interpolate(preds, size=image.shape[-2:], mode="bilinear", align_corners=False)

            elif task == 'segdepth':
                seg_logits, depth_preds = model(image)
                upsampled_seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                seg_preds = torch.softmax(upsampled_seg_logits, dim=1)
                seg_preds = torch.argmax(seg_preds, dim=1)
                
                depth_preds = F.relu(depth_preds, inplace=True)
                depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)

            else:
                raise ValueError(f'Unknown task: {task}')

            ender.record()

            torch.cuda.synchronize()

            current_time = starter.elapsed_time(ender)
            times.append(current_time)

    return mean(times), stdev(times)


'''
Generator that Predicts with the model
(Yields 5 tensors: image, label, depth, seg_preds, depth_preds)
'''
def predict(model, range, task):

    dataset = NYUv2Dataset(split='test')

    tensors = []

    # Prediction
    with torch.no_grad():
        for image in tqdm(range):

            image, label, depth = dataset[image]
            image = image.unsqueeze(0).to(config.DEVICES[0])

            if task == 'seg':
                seg_logits = model(image)
                upsampled_seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                seg_preds = torch.softmax(upsampled_seg_logits, dim=1)
                seg_preds = torch.argmax(seg_preds, dim=1)
                seg_preds = seg_preds.squeeze()
                depth_preds = None

            elif task == 'depth':
                depth_preds = model(image)
                depth_preds = F.relu(depth_preds, inplace=True)
                depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)
                depth_preds = depth_preds.squeeze()
                seg_preds = None

            elif task == 'segdepth':
                seg_logits, depth_preds = model(image)
                upsampled_seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                seg_preds = torch.softmax(upsampled_seg_logits, dim=1)
                seg_preds = torch.argmax(seg_preds, dim=1)
                seg_preds = seg_preds.squeeze()
                
                depth_preds = F.relu(depth_preds, inplace=True)
                depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)
                depth_preds = depth_preds.squeeze()

            else:
                raise ValueError(f'Unknown task: {task}')
            
            image = image.squeeze()
            label = label.squeeze()
            depth = depth.squeeze()

            yield image, label, depth, seg_preds, depth_preds
            
        

