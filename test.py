import transformers
import torch
from model import SegFormer, DepthFormer, SegDepthFormer
import config
from test_utils import evaluate_inference_time

'''
Evaluates the inference time of a model
'''
if __name__ == '__main__':

    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the model
    if config.TASK == 'seg':
        model = SegFormer()
    elif config.TASK == 'depth':
        model = DepthFormer()
    elif config.TASK == 'segdepth':
        model = SegDepthFormer()
    else:
        raise ValueError(f'Unknown task: {config.TASK}')
    
    # Load the model
    checkpoint = torch.load(config.CHECKPOINT, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    time_mean, time_std = evaluate_inference_time(model, repetitions=100, task=config.TASK)
    print(f'Mean inference time: {time_mean} ms')
    print(f'STD inference time: {time_std} ms')