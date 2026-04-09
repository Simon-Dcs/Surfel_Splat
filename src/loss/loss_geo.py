from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch
import numpy as np
import random

@dataclass
class LossGeoCfg:
    weight: float
    apply_after_step: int
    l1_weight: float = 0.7  # L1 term weight for edge preservation.
    l2_weight: float = 0.3  # L2 term weight for smoothing.
    use_si_loss: bool = False  # Enable the scale-invariant depth term.
    si_weight: float = 0.0  # Scale-invariant term weight.


@dataclass
class LossGeoCfgWrapper:
    geo: LossGeoCfg


class LossGeo(Loss[LossGeoCfg, LossGeoCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        """
        Depth loss function - L1+L2 hybrid

        L1 loss: better for edges and details, more robust
        L2 loss: smoothing effect, faster convergence
        Scale-Invariant loss: robust to depth scale changes (optional)
        """
        pred_depth = gaussians.depth  # (b, v, 1, h, w)
        target_depth = batch["context"]["depth"] / 200.0  # Rescale to the working depth range.

        # Create mask: filter out zero ground truth and excessive depth values
        mask = ((target_depth >= 0.5) & (target_depth <= 4.53)).float()
        valid_pixels = mask.sum()

        # Avoid division by zero
        if valid_pixels == 0:
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

        # ========================================
        # L1 loss (primary) - edge preservation
        # ========================================
        l1_diff = torch.abs(pred_depth - target_depth) * mask
        l1_loss = l1_diff.sum() / valid_pixels

        # ========================================
        # L2 loss (auxiliary) - smoothing
        # ========================================
        l2_diff = ((pred_depth - target_depth) ** 2) * mask
        l2_loss = l2_diff.sum() / valid_pixels

        # ========================================
        # Combine L1 and L2 losses
        # ========================================
        combined_loss = (
            self.cfg.l1_weight * l1_loss +
            self.cfg.l2_weight * l2_loss
        )

        # ========================================
        # Scale-Invariant loss (optional)
        # ========================================
        if self.cfg.use_si_loss and self.cfg.si_weight > 0:
            eps = 1e-6
            # Compute in log space
            log_pred = torch.log(pred_depth.clamp(min=eps))
            log_target = torch.log(target_depth.clamp(min=eps))
            log_diff = (log_pred - log_target) * mask

            # Scale-invariant term
            si_loss = (
                (log_diff ** 2).sum() / valid_pixels -
                (log_diff.sum() ** 2) / (valid_pixels ** 2)
            )

            combined_loss = combined_loss + self.cfg.si_weight * si_loss

        return self.cfg.weight * combined_loss

# class LossGeo(Loss[LossGeoCfg, LossGeoCfgWrapper]):
#     def forward(
#         self,
#         prediction: DecoderOutput,
#         batch: BatchedExample,
#         gaussians: Gaussians,
#         global_step: int,
#         #dump: dict,
#     ) -> Float[Tensor, ""]:
#         near = batch["target"]["near"][..., None, None]
#         far = batch["target"]["far"][..., None, None]

#         #print('prediction.depth mean is ',prediction.depth.max())
#         # np.save('check/'+str(global_step)+'_predict.npy',prediction.depth.cpu().detach().numpy())
#         # np.save('check/'+str(global_step)+'_color.npy',prediction.color.cpu().detach().numpy())
#         #print('batch["target"]["depth"] mean is ',np.max(batch["target"]["depth"].cpu().detach().numpy()))
#         #far = prediction.depth.max()
#         depth = (1000 * prediction.depth).minimum(far).maximum(near)
#         # print('depth prediction mean is ',depth.mean())
#         predict_depth = (depth - near) / (far - near)

#         #np.save('check/'+str(global_step)+'_apredict.npy',predict_depth.cpu().detach().numpy())

#         #far = batch["target"]["depth"].max()
#         gt_depth = batch["target"]["depth"].minimum(far).maximum(near)
#         # print('depth GT mean is ',gt_depth.mean())
#         gt_depth = (gt_depth - near) / (far - near)

#         #np.save('check/'+str(global_step)+'_gt.npy',gt_depth.cpu().detach().numpy())

#         delta_depth = gt_depth - predict_depth
#         #print('depth depth mean is ',delta_depth.mean())
#         #np.save('check/'+str(global_step)+'_delta.npy',delta_depth.cpu().detach().numpy())

#         # Create a mask that filters the minimum value.
#         # For each pixel, determine whether it is the minimum.
#         min_depth = torch.min(gt_depth, dim=-1, keepdim=True)[0]
#         min_depth = torch.min(min_depth, dim=-2, keepdim=True)[0]
#         # Mask value 1 marks a non-minimum position.
#         mask = (gt_depth > min_depth).float()

#         # np.save('check/mask.npy',mask.cpu().detach().numpy())
#         # Only compute loss on non-minimum positions.
#         depth_loss = (delta_depth**2) * mask
#         #print('depth_loss mean is ',depth_loss.mean())
#         # Only use non-minimum positions when averaging the loss.
#         valid_pixels = torch.sum(mask)
#         if valid_pixels > 0:
#             return self.cfg.weight * torch.sum(depth_loss) / valid_pixels
#         return torch.tensor(0.0, device=depth_loss.device)

# def gau_val(x,mean,cor):
#     return  torch.exp(-1/2 * (x-mean)@ torch.linalg.inv(cor)@(x-mean).t())[0]   




# class LossGeo(Loss[LossGeoCfg, LossGeoCfgWrapper]):
#     def forward(
#         self,
#         prediction: DecoderOutput,
#         batch: BatchedExample,
#         gaussians: Gaussians,
#         global_step: int,
#         #dump: dict,
#     ) -> Float[Tensor, ""]:
#         k = 4
#         L = 5000
#         loss = 0
#         batch_size, n_gaussian, dim = gaussians.means.shape
#         device = gaussians.means.device
#         for batch_idx in range(batch_size):
#             current_means = gaussians.means[batch_idx]
#             distances = torch.cdist(current_means, current_means)
#             distances.fill_diagonal_(float('inf'))
#             _, nearest_indices = torch.topk(distances, k, dim=1, largest=False) 
#             for gaussian_idx in random.sample(range(n_gaussian),L):
#                 nearest_means = current_means[nearest_indices[gaussian_idx]]  # shape: [k, dim]
#                 near_cor = gaussians.covariances[batch_idx][nearest_indices[gaussian_idx]] # shape: [k, dim, dim]
#                 current_mean = current_means[gaussian_idx].unsqueeze(0)  # shape: [1, dim]
#                 cur_cor = gaussians.covariances[batch_idx][gaussian_idx] # shape: [dim, dim]
#                 for i in range(k):
#                     new_mean = (nearest_means[i] + current_mean[0]) / 2
#                     gau_value = gau_val(new_mean.unsqueeze(0),nearest_means[i].unsqueeze(0),near_cor[i]) 
#                     gau_value += gau_val(new_mean.unsqueeze(0),current_mean,cur_cor) 
#                     gaussian_val = torch.minimum(gau_value, 1)
#                     loss += 1 - gaussian_val

#         return loss / k / L / batch_size



# def interpolate_nearest_means(means: torch.Tensor, k: int = 4, interpolation_weights: torch.Tensor = None) -> torch.Tensor:
#     """
#     For each mean in every batch, find the k nearest neighbors and interpolate.
    
#     Args:
#         means: tensor with shape [batch, gaussian, dim]
#         k: number of nearest neighbors per mean
#         interpolation_weights: optional interpolation weights with shape [k]
        
#     Returns:
#         interpolated_points: tensor with shape [batch, gaussian, k, dim]
#     """
#     batch_size, n_gaussian, dim = means.shape
#     device = means.device
    
#     # Initialize the output tensor.
#     interpolated_points = torch.zeros(batch_size, n_gaussian, k, dim, device=device)
    
#     # Use uniform weights when none are provided.
#     if interpolation_weights is None:
#         interpolation_weights = torch.ones(k, device=device) / k
    
#     # Process each batch independently.
#     for batch_idx in range(batch_size):
#         # Get the means for the current batch.
#         current_means = means[batch_idx]  # shape: [gaussian, dim]
        
#         # Compute pairwise Euclidean distances between means.
#         distances = torch.cdist(current_means, current_means)  # shape: [gaussian, gaussian]
        
#         # Ignore self-distance so a point is not selected as its own neighbor.
#         distances.fill_diagonal_(float('inf'))
        
#         # Find the indices of the k nearest neighbors for each mean.
#         _, nearest_indices = torch.topk(distances, k, dim=1, largest=False)  # shape: [gaussian, k]
        
#         # Process each mean.
#         for gaussian_idx in range(n_gaussian):
#             # Get the k nearest neighbors for the current mean.
#             nearest_means = current_means[nearest_indices[gaussian_idx]]  # shape: [k, dim]
#             current_mean = current_means[gaussian_idx].unsqueeze(0)  # shape: [1, dim]
            
#             # Compute interpolated points between the current mean and its neighbors.
#             for i in range(k):
#                 weight = interpolation_weights[i]
#                 interpolated_points[batch_idx, gaussian_idx, i] = (
#                     weight * current_mean + (1 - weight) * nearest_means[i]
#                 )
    
#     return interpolated_points
