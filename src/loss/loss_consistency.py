from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch

@dataclass
class LossConsistencyCfg:
    weight: float


@dataclass
class LossConsistencyCfgWrapper:
    consist: LossConsistencyCfg



class LossConsistency(Loss[LossConsistencyCfg, LossConsistencyCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:

        return self.cfg.weight * delta.mean()
