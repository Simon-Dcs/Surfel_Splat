from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch
from einops import rearrange, repeat

@dataclass
class LossNormalCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossNormalCfgWrapper:
    normal: LossNormalCfg


class LossNormal(Loss[LossNormalCfg, LossNormalCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        #dump: dict,
    ) -> Float[Tensor, ""]:
        normal = prediction.normal
        #normal = (normal + 1)/2

        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=normal.device)

        target_normal = 2 * batch["target"]["normal"] - 1

        # Create mask: positions where target normal is not all zeros
        mask = (target_normal.abs().sum(dim=2, keepdim=True) > 0).float()  # [b,v,1,h,w]

        delta = normal - target_normal
        masked_delta = (delta**2) * mask

        # Compute average only on valid pixels
        valid_pixels = mask.sum()
        if valid_pixels > 0:
            loss = masked_delta.sum() / valid_pixels
        else:
            loss = torch.tensor(0.0, device=normal.device)

        return self.cfg.weight * loss
    


        # delta = prediction.color - batch["target"]["image"]
        # #pred_depth = prediction.depth
        # pred_depth = gaussians.depth
        # #print(pred_depth.shape)
        # target_depth = batch["context"]["depth"] #/ 200
        # #print(target_depth.shape)
        # b,v,s,h,w = target_depth.shape
        # pred_depth = rearrange(pred_depth,"b (v h w) s -> b v s h w",v=v,h=h,w=w)
        # # Create a mask for positions where the ground truth is non-zero.
        # mask = (target_depth != 0).float()
        
        # # Compute the mean squared error under the mask.
        # depth_delta = (pred_depth - target_depth)**2
        # masked_depth_delta = depth_delta * mask
        
        # # Average over the valid pixels.
        # valid_pixels = mask.sum()
        # if valid_pixels > 0:
        #     depth_loss = masked_depth_delta.sum() / valid_pixels
        # else:
        #     depth_loss = torch.tensor(0.0, device=pred_depth.device)
        
        # # if depth_loss > 10.0:
        # #     depth_loss = torch.tensor(0.0, device=pred_depth.device)
        # #     print('skip bad sample')

        # return self.cfg.weight * depth_loss
