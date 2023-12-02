import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from dataset import NYUv2DataModule
from model import SegFormer, DepthFormer, SegDepthFormer
import transformers
from pytorch_lightning.callbacks import LearningRateMonitor

"""
Trains the model
"""

if __name__ == '__main__':
    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the logger
    logger = TensorBoardLogger('logs', name=config.NAME, version=config.VERSION)

    # Initialize the data module
    data_module = NYUv2DataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the model
    if config.TASK == 'seg':
        model = SegFormer()
    elif config.TASK == 'depth':
        model = DepthFormer()
    elif config.TASK == 'segdepth':
        model = SegDepthFormer()

    # Erstellen Sie eine Instanz von LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='gpu', precision=config.PRECISION, devices=config.DEVICES, callbacks=[lr_monitor])

    # Train the model
    trainer.fit(model, data_module, ckpt_path=config.CHECKPOINT)