import transformers
import torch
from model import SegFormer, DepthFormer, SegDepthFormer
import config
from test_utils import evaluate_inference_time, predict
from plot_utils import visualize_img_label, visualize_img_depth, visualize_img_label_depth, visualize_img_gts

'''
Evaluates the inference time of a model or predicts with the model
'''
if __name__ == '__main__':

    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the model
    if config.TASK == 'seg':
        model = SegFormer().to(config.DEVICES[0])
    elif config.TASK == 'depth':
        model = DepthFormer().to(config.DEVICES[0])
    elif config.TASK == 'segdepth':
        model = SegDepthFormer().to(config.DEVICES[0])
    else:
        raise ValueError(f'Unknown task: {config.TASK}')
    
    # Load the model
    checkpoint = torch.load(config.CHECKPOINT, map_location=torch.device(config.DEVICES[0]))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    '''
    # Evaluate the inference time
    time_mean, time_std = evaluate_inference_time(model, repetitions=100000, task=config.TASK)
    print(f'Mean inference time: {time_mean} ms')
    print(f'STD inference time: {time_std} ms')

    # Append the results to the results file
    with open('results_inf_time.txt', 'a') as f:
        f.write(f'{config.TASK}: {time_mean} ms, {time_std} ms\n')
    '''
    
    #'''
    # Predict with the model
    model.eval()
    # Use Generator to predict
    cnt = 0
    for image, label, depth, seg_preds, depth_preds in predict(model, 500, config.TASK):
        if config.TASK == 'seg':
            visualize_img_label(image.cpu(), label.cpu(), seg_preds.cpu(), filename=f'seg/seg_{cnt}.png')
        elif config.TASK == 'depth':
            visualize_img_depth(image.cpu(), depth.cpu(), depth_preds.cpu(), filename=f'depth/depth_{cnt}.png')
        elif config.TASK == 'segdepth':
            visualize_img_label_depth(image.cpu(), label.cpu(), seg_preds.cpu(), depth.cpu(), depth_preds.cpu(), filename=f'segdepth/segdepth_{cnt}.png')

        cnt += 1
    #'''