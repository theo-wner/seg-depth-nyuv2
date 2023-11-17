from statistics import mean, stdev
import torch
import torch.nn.functional as F
from dataset import NYUv2Dataset


'''
Evaluates the inference time of a model
'''
def evaluate_inference_time(model, repetitions, task):

    dataset = NYUv2Dataset(split='test')
    image, _, _ = dataset[0]
    image = image.unsqueeze(0).cuda()

    # inference time measurement
    times = []
    with torch.no_grad():
        for repetition in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            starter.record()
            
            if task == 'seg':
                loss, logits = model(image)
                logits = torch.nn.functional.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

                probability_map = torch.softmax(logits, dim=1)
                prediction_map = torch.argmax(probability_map, dim=1)
                entropy_map = -torch.sum(probability_map * torch.log(probability_map + 1e-6), dim=1)

            elif task == 'depth':
                logits = model(image)[0]
                depth = F.relu(logits[:, 0, :, :].unsqueeze(dim=1))
                variance = F.softplus(logits[:, 1, :, :].unsqueeze(dim=1))
                depth = torch.nn.functional.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)
                variance = torch.nn.functional.interpolate(variance, size=image.shape[-2:], mode="bilinear", align_corners=False)

            elif task == 'segdepth':
                segmentation_logits, depth_logits, variance_logits = model(image)
                segmentation_logits = torch.nn.functional.interpolate(segmentation_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                depth_logits = torch.nn.functional.interpolate(depth_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
                variance_logits = torch.nn.functional.interpolate(variance_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

                probability_map = torch.softmax(segmentation_logits, dim=1)
                prediction_map = torch.argmax(probability_map, dim=1)
                entropy_map = -torch.sum(probability_map * torch.log(probability_map + 1e-6), dim=1)

            else:
                raise ValueError(f'Unknown task: {task}')

            ender.record()

            torch.cuda.synchronize()

            current_time = starter.elapsed_time(ender)
            times.append(current_time)

    return mean(times), stdev(times)