from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        #dump: dict,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        
        delta = prediction.color - image
        #print('delta.shape is',delta.shape)
        return self.cfg.weight * (delta**2).mean()
