import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import wandb
from torch.utils.data import DataLoader, Dataset
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

from logging_setup import get_logger

logger = get_logger()
torch.set_float32_matmul_precision('medium')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


class SimpleDataset(Dataset):
    def __init__(self, inputs, labels, transform=None, preprocess=None):
        """
        Args:
            inputs (list or ndarray): The input features.
            labels (list or ndarray): The labels corresponding to the inputs.
            transform (callable, optional): Optional transform to be applied on a sample.
            preprocess (callable, optional): Optional preprocessing to be applied on a sample.
        """

        # If there's a preprocessing function, apply it
        if preprocess is not None:
            inputs = preprocess(inputs)

        self.inputs = inputs
        self.labels = labels
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = torch.tensor(self.inputs[idx], dtype=DTYPE)
        label = torch.tensor(self.labels[idx], dtype=DTYPE).squeeze()  # Ensure labels are 1D

        if self.transform is not None:
            image = self.transform(image)

        return image, label


### Lightning Module ###
class ZooBot_lightning(pl.LightningModule):
    def __init__(self, zoobot_size, zoobot_blocks, learning_rate, learning_decay):
        super(ZooBot_lightning, self).__init__()
        self.save_hyperparameters()  # Saves all arguments for checkpointing

        # Define the model
        self.model = FinetuneableZoobotClassifier(
            name=f'hf_hub:mwalmsley/zoobot-encoder-convnext_{zoobot_size}',
            n_blocks=zoobot_blocks,  # Finetune this many blocks.x
            learning_rate=learning_rate,  # use a low learning rate
            lr_decay=learning_decay,  # reduce the learning rate from lr to lr^0.5 for deeper blocks
            num_classes=2,  # Number of output classes
        )

        self.learning_rate = learning_rate

        self.train_step_outputs = []
        self.valid_step_outputs = []

        self.validation_outputs = []
        self.validation_targets = []

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Get model outputs
        outputs = self(images)

        # Convert single probability labels to two-class probabilities
        # If label is 0.7, target becomes [0.3, 0.7]
        target = torch.stack([1 - labels, labels], dim=1)

        # Apply log_softmax to model outputs
        log_probabilities = F.log_softmax(outputs, dim=1)

        # Calculate KL divergence loss
        # reduction='batchmean' gives us the average loss per batch
        loss = F.kl_div(log_probabilities, target, reduction='batchmean')

        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # Calculate probabilities
        probabilities = F.softmax(outputs, dim=1)
        positive_class_probs = probabilities[:, 1]  # Dwarf probability

        # Create target probability distribution
        target_probs = torch.stack([1 - labels, labels], dim=1)

        # 1. KL Divergence Loss
        log_probabilities = F.log_softmax(outputs, dim=1)
        kl_loss = F.kl_div(log_probabilities, target_probs, reduction='batchmean')

        # 2. Brier Score (MSE)
        brier_score = F.mse_loss(positive_class_probs, labels)

        # 3. Expected Calibration Error
        calibration_error = tm.functional.calibration_error(
            positive_class_probs,
            (labels >= 0.5).long(),  # Threshold for class alignment
            n_bins=10,
            norm='l1',
            task='binary',  # Add this parameter
        )

        # Store outputs and labels for visualization
        self.validation_outputs.append(positive_class_probs.detach().cpu().numpy())
        self.validation_targets.append(labels.detach().cpu().numpy())

        # Store the main loss for epoch end calculations
        self.valid_step_outputs.append(kl_loss.item())

        # Log all metrics
        self.log_dict(
            {
                'valid_loss': kl_loss,
                'val_brier': brier_score,
                'val_calibration': calibration_error,
            },
            prog_bar=True,
        )

        return kl_loss

    def on_train_epoch_end(self):
        train_loss = sum(self.train_step_outputs) / len(self.train_step_outputs)
        self.log('train_loss', train_loss)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        valid_loss = sum(self.valid_step_outputs) / len(self.valid_step_outputs)
        self.log('valid_loss', valid_loss)
        self.valid_step_outputs.clear()

        # Concatenate stored outputs and targets
        outputs = np.concatenate(self.validation_outputs)
        targets = np.concatenate(self.validation_targets)

        # 1. Predictions vs True Values
        fig1 = plt.figure(figsize=(6, 6))
        h = plt.hist2d(
            targets, outputs, bins=[25, 25], range=[[0, 1], [0, 1]], cmap='viridis', norm='log'
        )
        plt.plot([0, 1], [0, 1], c='r', linestyle='--')
        plt.axhline(targets.mean(), linestyle='--', color='b')
        plt.colorbar(h[3])
        plt.ylabel('Model Predictions')
        plt.xlabel('Expert Classifications')
        plt.title('Prediction vs Truth Distribution')
        plt.tight_layout()
        self.logger.experiment.log({'Prediction Distribution': wandb.Image(fig1)})
        plt.close(fig1)

        # 2. Reliability (Calibration) Curve
        fig2 = plt.figure(figsize=(6, 6))
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(outputs, bin_edges) - 1

        mean_predicted = np.zeros(n_bins)
        mean_true = np.zeros(n_bins)

        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                mean_predicted[i] = outputs[bin_mask].mean()
                mean_true[i] = targets[bin_mask].mean()

        plt.plot([0, 1], [0, 1], 'r--')
        plt.plot(mean_predicted, mean_true, 'bo-')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Mean True Probability')
        plt.title('Reliability Curve')
        plt.tight_layout()
        self.logger.experiment.log({'Reliability Curve': wandb.Image(fig2)})
        plt.close(fig2)

        # 3. Error Distribution
        fig3 = plt.figure(figsize=(6, 6))
        errors = outputs - targets
        plt.hist(errors, bins=50, density=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.tight_layout()
        self.logger.experiment.log({'Error Distribution': wandb.Image(fig3)})
        plt.close(fig3)

        # Clear stored data
        self.validation_outputs.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        # 1. Define your optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # 2. Define the scheduler
        # Here we monitor 'train_loss' (but you can also monitor 'val_loss'
        # if you prefer to reduce LR based on validation performance).
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # we want to minimize loss
            factor=0.5,  # multiply LR by 0.1 each time the metric plateaus
            patience=5,  # wait 'patience' epochs with no improvement
            min_lr=1e-6,  # optional: set a lower bound
            verbose=True,  # to print LR reductions in logs
        )

        # 3. Return both optimizer and scheduler with a 'monitor' key
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}


class AddBimodalNoise(object):
    def __init__(self, means, stds=None, weights=None, noise_scale=0.1):
        """
        Args:
            means: List of two means [mean1, mean2]
            stds: List of two stds [std1, std2]
            weights: List of two weights [w1, w2]
            noise_scale: Scale factor for noise (default 0.1)
        """
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds if stds else [0.1, 0.1])
        self.weights = torch.tensor(weights if weights else [0.5, 0.5])
        self.noise_scale = noise_scale

    def __call__(self, tensor):
        """
        Add noise while preserving bimodal distribution.
        """
        # Calculate distances to each mode
        dist1 = torch.abs(tensor - self.means[0])
        dist2 = torch.abs(tensor - self.means[1])

        # Simple mask based on which mode is closer
        mask = dist1 < dist2

        # Generate noise for each mode
        noise1 = torch.randn_like(tensor) * self.stds[0] * self.noise_scale
        noise2 = torch.randn_like(tensor) * self.stds[1] * self.noise_scale

        # Apply noise based on mask
        noise = torch.where(mask, noise1, noise2)

        # Ensure no NaN or inf values
        noise = torch.nan_to_num(noise, 0.0)

        return torch.clamp(tensor + noise, -1.0, 1.0)


def get_dwarf_predictions(model, data, batch_size=128, dtype=torch.float32, device='cpu'):
    """
    Run inference on preprocessed data to get dwarf predictions.

    Args:
        model: Loaded and evaluated ZooBot model
        data (np.ndarray): Preprocessed image data
        batch_size (int): Batch size for inference
        dtype: Torch data type
        device: Device to run inference on

    Returns:
        np.ndarray: Array of dwarf probabilities for each input
    """
    # Verify input data type
    assert data.dtype == np.float32, 'Input must be float32'

    predictions = []

    # Create dataloader for batched inference
    dataset = SimpleDataset(data, np.zeros(len(data)))  # Dummy labels
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch, _ in dataloader:
            # Move batch to correct device and type
            batch = batch.to(device=device, dtype=dtype)

            # Get model predictions
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)
            positive_class_probs = probabilities[:, 1]

            # Store predictions
            predictions.append(positive_class_probs.cpu().numpy())

    return np.concatenate(predictions)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    model = ZooBot_lightning.load_from_checkpoint(model_path, **hparams)
    state_dict = model.state_dict()
    logger.info('Sucessfully loaded model from checkpoint.')
    return state_dict, hparams
