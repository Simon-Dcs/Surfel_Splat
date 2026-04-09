from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch

@dataclass
class LossDistCfg:
    weight: float


@dataclass
class LossDistCfgWrapper:
    dist: LossDistCfg


class LossDist(Loss[LossDistCfg, LossDistCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        #dump: dict,
    ) -> Float[Tensor, ""]:
        #print('working')
        return self.cfg.weight * prediction.distortion.mean()