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
        model = SegFormer.load_from_checkpoint(config.CHECKPOINT)
        model = model.to(config.DEVICES[0])
    elif config.TASK == 'depth':
        model = DepthFormer()
        model = DepthFormer.load_from_checkpoint(config.CHECKPOINT)
        model = model.to(config.DEVICES[0])
    elif config.TASK == 'segdepth':
        model = SegDepthFormer()
        model = SegDepthFormer.load_from_checkpoint(config.CHECKPOINT)
        model = model.to(config.DEVICES[0])
    else:
        raise ValueError(f'Unknown task: {config.TASK}')
    
    time_mean, time_std = evaluate_inference_time(model, repetitions=100, task=config.TASK)
    print(f'Mean inference time: {time_mean} ms')
    print(f'STD inference time: {time_std} ms')