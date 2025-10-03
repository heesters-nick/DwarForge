import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
import wandb
from torch.utils.data import DataLoader, Dataset
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
EPSILON = 1e-8

### Color blind palette ###
colors = {
    'blue': '#377eb8',
    'orange': '#ff7f00',
    'green': '#4daf4a',
    'pink': '#f781bf',
    'brown': '#a65628',
    'purple': '#984ea3',
    'gray': '#999999',
    'red': '#e41a1c',
    'yellow': '#dede00',
}


class SimpleDataset_v1(Dataset):
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


class SimpleDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32)


class ZooBot_lightning_v1(pl.LightningModule):
    def __init__(self, zoobot_size, zoobot_blocks, learning_rate, learning_decay):
        super(ZooBot_lightning_v1, self).__init__()
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
        self.logger.experiment.log({'Prediction Distribution': wandb.Image(fig1)})  # type: ignore
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
        self.logger.experiment.log({'Reliability Curve': wandb.Image(fig2)})  # type: ignore
        plt.close(fig2)

        # 3. Error Distribution
        fig3 = plt.figure(figsize=(6, 6))
        errors = outputs - targets
        plt.hist(errors, bins=50, density=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.tight_layout()
        self.logger.experiment.log({'Error Distribution': wandb.Image(fig3)})  # type: ignore
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
        )

        # 3. Return both optimizer and scheduler with a 'monitor' key
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}


### Lightning Module ###
class ZooBot_lightning(pl.LightningModule):
    def __init__(
        self,
        zoobot_size,
        zoobot_blocks,
        learning_rate,
        learning_decay,
        weight_decay,
        label_smoothing,
        loss_type='kld',
        focal_gamma=2.0,
    ):
        super(ZooBot_lightning, self).__init__()
        self.save_hyperparameters()  # Saves all arguments for checkpointing

        if self.hparams.loss_type not in ['focal', 'kld']:  # type: ignore
            raise ValueError(
                f"Invalid loss_type: {self.hparams.loss_type}. Choose 'focal' or 'kld'."  # type: ignore
            )

        # Define the model
        self.model = FinetuneableZoobotClassifier(
            name=f'hf_hub:mwalmsley/zoobot-encoder-convnext_{zoobot_size}',
            n_blocks=zoobot_blocks,  # Finetune this many blocks.x
            learning_rate=learning_rate,  # use a low learning rate
            lr_decay=learning_decay,  # reduce the learning rate from lr to lr^0.5 for deeper blocks
            num_classes=2,  # Number of output classes
        )

        self.train_step_outputs = []
        self.valid_step_outputs = []

        self.validation_outputs = []
        self.validation_targets = []

    def forward(self, x):
        x = self.model(x)
        return x

    def _compute_focal_loss(self, probabilities, targets):
        """
        Computes the Focal loss.
        Args:
            probabilities: Predicted probabilities (shape: [batch_size, 2]).
            targets: True labels (shape: [batch_size]).
        Returns:
            Mean Focal loss for the batch.
        """
        gamma = self.hparams.focal_gamma  # type: ignore
        p_pred = probabilities[:, 1]
        p_pred_clamped = p_pred.clamp(min=EPSILON, max=1.0 - EPSILON)
        loss_positive = -targets * torch.pow(1.0 - p_pred, gamma) * torch.log(p_pred_clamped)
        loss_negative = (
            -(1.0 - targets) * torch.pow(p_pred, gamma) * torch.log(1.0 - p_pred_clamped)
        )
        loss_unreduced = loss_positive + loss_negative
        return loss_unreduced.mean()

    def _compute_kld_loss(self, logits, targets):
        """
        Computes the KL Divergence loss.
        Args:
            logits: Raw logits output from the model (shape: [batch_size, 2]).
            targets: Soft labels (probabilities for positive class) (shape: [batch_size]).
        Returns:
            Mean KL divergence loss for the batch.
        """
        # Create the target probability distribution [P(class 0), P(class 1)]
        target_dist = torch.stack([1.0 - targets, targets], dim=1)

        # Calculate log probabilities from logits
        log_probabilities = F.log_softmax(logits, dim=1)

        # Calculate KL divergence
        # Use log_target=False because target_dist contains probabilities, not log-probabilities
        loss = F.kl_div(log_probabilities, target_dist, reduction='batchmean', log_target=False)
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch

        p_true = self.apply_label_smoothing(labels, alpha=self.hparams.label_smoothing)  # type: ignore

        # Get model outputs
        logits = self(images)  # Logits shape [batch, 2]

        if self.hparams.loss_type == 'focal':  # type: ignore
            probabilities = F.softmax(logits, dim=1)
            loss = self._compute_focal_loss(probabilities, p_true)
        elif self.hparams.loss_type == 'kld':  # type: ignore
            loss = self._compute_kld_loss(logits, p_true)
        else:
            # This should not happen due to check in __init__ but good practice
            raise ValueError(f'Invalid loss_type specified: {self.hparams.loss_type}')  # type: ignore

        self.train_step_outputs.append(loss)
        return loss

    def apply_label_smoothing(self, labels, alpha=None):
        """Apply label smoothing with strength alpha."""
        # Move probabilities slightly toward 0.5
        if alpha is None or alpha == 0.0:
            return labels
        else:
            return labels * (1 - alpha) + 0.5 * alpha

    def expected_calibration_error(self, preds, soft_labels, n_bins=5):
        """
        Calculate ECE for soft labels.

        Args:
            preds: Predicted probabilities (tensor)
            soft_labels: Soft label probabilities (tensor)
            n_bins: Number of bins to use

        Returns:
            ece: The Expected Calibration Error
            bin_data: Dictionary with detailed bin information
        """
        # Convert to numpy for easier manipulation
        preds_np = preds.detach().cpu().numpy()
        labels_np = soft_labels.detach().cpu().numpy()

        # Create equal-width bins across prediction range
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(preds_np, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases

        # Initialize arrays for bin statistics
        bin_sizes = np.zeros(n_bins)
        bin_avg_preds = np.zeros(n_bins)
        bin_avg_labels = np.zeros(n_bins)

        # Calculate statistics for each bin
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            bin_sizes[bin_idx] = mask.sum()

            if bin_sizes[bin_idx] > 0:
                bin_avg_preds[bin_idx] = preds_np[mask].mean()
                bin_avg_labels[bin_idx] = labels_np[mask].mean()

        # Calculate ECE with proper weighting
        total_samples = len(preds_np)
        ece = np.sum((bin_sizes / total_samples) * np.abs(bin_avg_preds - bin_avg_labels))

        # Return ECE and bin data for visualizations
        bin_data = {
            'sizes': bin_sizes,
            'avg_preds': bin_avg_preds,
            'avg_labels': bin_avg_labels,
            'boundaries': bin_boundaries,
        }

        return ece, bin_data

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        # Calculate probabilities
        probabilities = F.softmax(logits, dim=1)
        positive_class_probs = probabilities[:, 1]  # Dwarf probability

        # 1. Calculate the loss
        if self.hparams.loss_type == 'focal':  # type: ignore
            # Use original labels (not smoothed) for validation loss
            val_loss = self._compute_focal_loss(probabilities, labels)
        elif self.hparams.loss_type == 'kld':  # type: ignore
            # Use original labels (not smoothed) for validation loss
            val_loss = self._compute_kld_loss(logits, labels)
        else:
            raise ValueError(f'Invalid loss_type specified: {self.hparams.loss_type}')  # type: ignore

        # 2. Brier Score (MSE)
        brier_score = F.mse_loss(positive_class_probs, labels)

        # 3. Calculate soft-label ECE
        ece, bin_data = self.expected_calibration_error(positive_class_probs, labels)

        # Store bin data for visualization
        if batch_idx == 0:  # Only store once per epoch
            self.bin_data = bin_data

        # Store outputs and labels for visualization
        self.validation_outputs.append(positive_class_probs.detach().cpu().numpy())
        self.validation_targets.append(labels.detach().cpu().numpy())

        # Store the main loss for epoch end calculations
        self.valid_step_outputs.append(val_loss.item())

        # Log all metrics
        self.log_dict(
            {
                'valid_loss': val_loss,  # type: ignore
                'val_brier': brier_score,
                'val_calibration': ece,
            },
            prog_bar=True,
        )

        return val_loss

    def on_train_epoch_end(self):
        # train_loss = sum(self.train_step_outputs) / len(self.train_step_outputs)
        train_loss = torch.stack(self.train_step_outputs).mean()
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
        self.logger.experiment.log({'Prediction Distribution': wandb.Image(fig1)})  # type: ignore
        plt.close(fig1)

        # 2. Reliability (Calibration) Curve
        n_bins = 5  # Number of bins for reliability plot
        bin_edges = np.linspace(0, 1, n_bins + 1)

        # Bin data based on PREDICTED probabilities (outputs)
        bin_indices = np.digitize(
            outputs, bin_edges[1:], right=False
        )  # bin_edges[1:] -> bins [0, n_bins-1]

        mean_predicted = np.zeros(n_bins)
        mean_true = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            bin_mask = bin_indices == i
            counts[i] = np.sum(bin_mask)
            if counts[i] > 0:
                mean_predicted[i] = outputs[bin_mask].mean()
                mean_true[i] = targets[bin_mask].mean()
            # else: means remain 0, counts remain 0

        # Filter out bins with zero counts to avoid plotting artifacts
        valid_bins_mask = counts > 0
        mean_predicted_valid = mean_predicted[valid_bins_mask]
        mean_true_valid = mean_true[valid_bins_mask]
        counts_valid = counts[valid_bins_mask]

        # Create the plot
        fig2 = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')  # Diagonal line

        if len(mean_predicted_valid) > 0:  # Only plot if there's valid data
            # Plot with conventional axes: Prediction (Confidence) on X, Truth (Accuracy) on Y
            plt.plot(mean_predicted_valid, mean_true_valid, 'bo-', label='Model Calibration')

            # Add sample count annotations
            for i in range(len(mean_predicted_valid)):
                # Adjust text position slightly for clarity
                plt.text(
                    mean_predicted_valid[i],
                    mean_true_valid[i] + 0.02,
                    f'n={counts_valid[i]}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
        else:
            plt.text(0.5, 0.5, 'No data in validation bins', ha='center', va='center')

        plt.xlabel('Mean Predicted Probability (Confidence)')  # Conventional X-axis label
        plt.ylabel('Mean True Label (Accuracy)')  # Conventional Y-axis label
        plt.title('Reliability Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim([-0.05, 1.05])  # Add padding
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        self.logger.experiment.log({'Reliability Curve': wandb.Image(fig2)})  # type: ignore
        plt.close(fig2)

        if hasattr(self, 'bin_data') and self.bin_data:
            avg_preds = self.bin_data['avg_preds']
            avg_labels = self.bin_data['avg_labels']
            sizes = self.bin_data['sizes']
            n_bins_bar = len(avg_preds)  # Get number of bins from data

            fig3 = plt.figure(figsize=(7, 6))  # Slightly wider for annotations
            bar_width = 0.35
            bin_indices_bar = np.arange(n_bins_bar)

            # Bars for average true labels
            plt.bar(
                bin_indices_bar - bar_width / 2,
                avg_labels,
                bar_width,
                label='Avg Label (Truth)',
                color=colors['blue'],
            )
            # Bars for average predictions
            plt.bar(
                bin_indices_bar + bar_width / 2,
                avg_preds,
                bar_width,
                label='Avg Prediction',
                color=colors['orange'],
            )

            # Add sample count annotations above bars
            for i in range(n_bins_bar):
                if sizes[i] > 0:  # Only annotate if count > 0
                    # Position text above the taller bar
                    y_pos = max(avg_labels[i], avg_preds[i]) + 0.02
                    plt.text(
                        bin_indices_bar[i],
                        y_pos,
                        f'n={int(sizes[i])}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                    )

            # Add ideal calibration line (y=x equivalent for bins)
            # Get bin boundaries for x-axis labels
            bin_boundaries = self.bin_data.get('boundaries', np.linspace(0, 1, n_bins_bar + 1))
            bin_centers_approx = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            plt.plot(
                bin_indices_bar, bin_centers_approx, 'r--', label='Perfect Calibration'
            )  # Plot against bin index

            # Get ECE value from logged metrics if available
            ece_metric_name = 'val_calibration'  # Make sure this matches the name used in validation_step log_dict
            ece_value = self.trainer.callback_metrics.get(ece_metric_name, None)
            title = 'Calibration Bar Chart'
            if ece_value is not None:
                title += f' (ECE = {float(ece_value):.4f})'  # Ensure ECE is float

            plt.xlabel('Prediction Bin Index')
            # Use bin boundaries for clearer x-axis ticks
            tick_labels = [
                f'{bin_boundaries[i]:.1f}-{bin_boundaries[i + 1]:.1f}' for i in range(n_bins_bar)
            ]
            plt.xticks(ticks=bin_indices_bar, labels=tick_labels, rotation=45, ha='right')
            plt.ylabel('Probability')
            plt.ylim([-0.05, 1.05])
            plt.title(title)
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            self.logger.experiment.log({'Calibration': wandb.Image(fig3)})  # type: ignore
            plt.close(fig3)

        # 3. Error Distribution
        fig4 = plt.figure(figsize=(6, 6))
        errors = outputs - targets
        plt.hist(errors, bins=50, density=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.tight_layout()
        self.logger.experiment.log({'Error Distribution': wandb.Image(fig4)})  # type: ignore
        plt.close(fig4)

        # Clear stored data
        self.validation_outputs.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and LR scheduler (ReduceLROnPlateau).

        Selects parameters for optimization based on self.hparams.zoobot_blocks,
        applying learning rate decay (self.hparams.learning_decay) to deeper blocks,
        mimicking the logic from FinetuneableZoobotAbstract.
        """
        # Retrieve hyperparameters
        try:
            lr = self.hparams.learning_rate  # type: ignore
            lr_decay_factor = self.hparams.learning_decay  # type: ignore
            num_blocks_to_tune = self.hparams.zoobot_blocks  # type: ignore
            # Use weight_decay from hparams
            weight_decay = self.hparams.weight_decay  # type: ignore
        except AttributeError as e:
            logger.error(
                f'Optimizer config failed: Missing hyperparameter. Ensure learning_rate, learning_decay, zoobot_blocks, (and optionally weight_decay) are saved. Error: {e}'
            )
            raise

        # Start parameter groups: always include the head (no LR decay)
        # Ensure self.model.head exists (it should be created by FinetuneableZoobotClassifier)
        if not hasattr(self.model, 'head'):
            raise AttributeError(
                "self.model does not have a 'head' attribute. Ensure FinetuneableZoobotClassifier initialization was successful."
            )

        params_to_optimize = [{'params': self.model.head.parameters(), 'lr': lr}]
        logger.info(f'Opt: Initializing Optimizer. Base LR: {lr}, Weight Decay: {weight_decay}')
        logger.info(f'Opt: Head parameters included with LR {lr}')

        if num_blocks_to_tune > 0:
            logger.info(f'Opt: Fine-tuning last {num_blocks_to_tune} encoder blocks/stages.')
            logger.info(f'Opt: Encoder architecture: {type(self.model.encoder).__name__}')

            # --- Parameter Group Selection Logic (adapted from FinetuneableZoobotAbstract) ---
            if isinstance(self.model.encoder, timm.models.ConvNeXt):
                # For ConvNeXt: stem + 4 stages
                tuneable_blocks_or_stages = [self.model.encoder.stem] + list(
                    self.model.encoder.stages
                )
                logger.info(
                    f'Opt: Identified {len(tuneable_blocks_or_stages)} tuneable blocks/stages for ConvNeXt (stem + stages).'
                )
            else:
                raise ValueError(
                    f'Opt: Encoder architecture {type(self.model.encoder).__name__} not explicitly handled in custom configure_optimizers.'
                )

            if num_blocks_to_tune > len(tuneable_blocks_or_stages):
                logger.info(
                    f'Opt: Requested {num_blocks_to_tune} blocks, but only {len(tuneable_blocks_or_stages)} available. Tuning all available.'
                )
                num_blocks_to_tune = len(tuneable_blocks_or_stages)

            # Reverse to order from last layer (highest index) to first
            tuneable_blocks_or_stages.reverse()
            blocks_to_tune = tuneable_blocks_or_stages[:num_blocks_to_tune]

            # Add parameter groups for encoder blocks with decayed LR
            for i, block in enumerate(blocks_to_tune):
                block_lr = lr * (
                    lr_decay_factor**i
                )  # Apply decay based on depth (i=0 is last block)
                block_params = list(block.parameters())
                if not block_params:
                    logger.info(
                        f'Opt: Block {i} (type {type(block).__name__}) has no parameters. Skipping.'
                    )
                    continue
                params_to_optimize.append({'params': block_params, 'lr': block_lr})
                logger.info(
                    f'Opt: Including block {i} (type {type(block).__name__}) with LR {block_lr:.2e}'
                )
            # --- End of Parameter Group Selection Logic ---
        else:
            logger.info('Opt: num_blocks_to_tune is 0. Only training the head.')

        logger.info(f'Opt: Total parameter groups for optimizer: {len(params_to_optimize)}')

        # 1. Define your chosen optimizer (AdamW)
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=lr,  # Base LR is default for AdamW, but groups override it
            weight_decay=weight_decay,
        )

        # 2. Define your chosen scheduler (ReduceLROnPlateau)
        # Monitoring 'valid_loss' which you log in validation_step
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Minimize the monitored metric (loss)
            factor=0.75,  # Reduce LR by half when plateaued
            patience=5,  # Number of epochs with no improvement to wait
            min_lr=1e-6,  # Minimum learning rate
        )

        # 3. Return configuration for PyTorch Lightning
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',  # Metric to monitor for scheduler
                'interval': 'epoch',  # Check metric at the end of each epoch
                'frequency': 1,  # Check every 1 epoch
            },
        }


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
    dataset = SimpleDataset_v1(data, np.zeros(len(data)))  # Dummy labels # type: ignore
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
    model = ZooBot_lightning_v1.load_from_checkpoint(model_path, **hparams)
    state_dict = model.state_dict()
    logger.info('Sucessfully loaded model from checkpoint.')
    return state_dict, hparams


def load_models(model_paths):
    """
    Loads all models for ensemble prediction.

    Args:
        model_paths (list): List of paths to model checkpoints

    Returns:
        list: List of loaded models
    """
    models = []

    for model_path in model_paths:
        try:
            logger.info(f'Loading model from {model_path}')

            # Load model state dict and hyperparameters
            checkpoint = torch.load(model_path, map_location='cpu')
            hparams = checkpoint['hyper_parameters']

            # Initialize model with hyperparameters
            model = ZooBot_lightning(**hparams)
            model.load_state_dict(checkpoint['state_dict'])
            model.freeze()
            model.eval()
            model = model.to(DEVICE)

            models.append(model)

        except Exception as e:
            logger.error(f'Error loading model from {model_path}: {e}')

    # sanity check
    test_model = models[0]
    logger.info(f'Models are on device: {next(test_model.parameters()).device}')
    logger.info(
        f'Models are in eval mode: {all(not p.requires_grad for p in test_model.parameters())}'
    )
    logger.info(f'Model architecture is: {test_model.model.encoder.default_cfg["architecture"]}')

    return models


def ensemble_predict(models, preprocessed_images, batch_size=64, device=None):
    """
    Runs inference using an ensemble of models and returns the average prediction.
    Optimized for both CPU and GPU processing.

    Args:
        models (list): List of loaded models
        preprocessed_images (np.ndarray): Batch of preprocessed images to predict on
        batch_size (int): Batch size for inference
        device (torch.device, optional): Device to run inference on. If None, uses current model device.

    Returns:
        np.ndarray: Array of mean predictions
    """
    # Determine the device to use
    if device is None:
        # Use the device of the first model if not specified
        device = next(models[0].parameters()).device

    # Configure DataLoader based on device
    pin_memory = device.type == 'cuda'

    # Create dataset and dataloader
    dataset = SimpleDataset(np.array(preprocessed_images))
    dataloader = DataLoader(
        dataset,
        batch_size=len(preprocessed_images),  # Process entire tile in one batch
        shuffle=False,
        num_workers=0,  # Keep at 0 for multiprocessing compatibility
        pin_memory=pin_memory,
    )

    # Initialize predictions array for each model
    all_model_predictions = []

    # Get predictions from each model
    for model_idx, model in enumerate(models):
        # Ensure model is on the correct device
        model = model.to(device)

        model_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to the same device as the model
                batch = batch.to(device)

                # Forward pass
                logits = model(batch)

                # Get probabilities for the positive class
                probs = F.softmax(logits, dim=1)[:, 1]

                # Move predictions back to CPU for aggregation
                model_predictions.extend(probs.cpu().numpy())

        all_model_predictions.append(model_predictions)

    # Stack predictions and calculate mean across models
    all_preds = np.array(all_model_predictions)
    ensemble_predictions = np.mean(all_preds, axis=0)

    # Clear GPU cache if using CUDA to prevent memory leaks in loops
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ensemble_predictions
