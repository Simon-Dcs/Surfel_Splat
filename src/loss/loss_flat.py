from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch

DIM = 3

@dataclass
class LossFlatCfg:
    weight: float


@dataclass
class LossFlatCfgWrapper:
    flat: LossFlatCfg


# class LossFlat(Loss[LossFlatCfg,LossFlatCfgWrapper]):
#     def forward(
#         self,
#         prediction: DecoderOutput,
#         batch: BatchedExample,
#         gaussians: Gaussians,
#         global_step: int,
#         dump: dict,
#     )-> Float[Tensor, ""]:
#         #scale_tensor = torch.linalg.eigvals(gaussians.covariances).abs()
#         #scale = dump['scales'].abs()
#         scale = gaussians.scales
#         #print('scale ',scale.shape)
#         min_scale = torch.min(scale, dim=2)[0]
#         return self.cfg.weight * min_scale.mean()
    

# class LossFlat(Loss[LossFlatCfg, LossFlatCfgWrapper]):
#     def forward(
#         self,
#         prediction: DecoderOutput,
#         batch: BatchedExample,
#         gaussians: Gaussians,
#         global_step: int,
#         #dump: dict,
#     ) -> Float[Tensor, ""]:
#         # Compute eigenvalues of covariance matrix
#         # eigenvalues shape will be [batch, gaussian, 3], as each 3x3 covariance matrix has 3 eigenvalues
#         eigenvalues = torch.linalg.eigvals(gaussians.covariances).abs()
        
#         # Find minimum eigenvalue for each Gaussian
#         # min_eigenvalues shape will be [batch, gaussian]
#         min_eigenvalues = torch.min(eigenvalues, dim=-1)[0]
#         # print('I AM HERE')
#         # Compute average of minimum eigenvalues
#         return self.cfg.weight * min_eigenvalues.mean()

class LossFlat(Loss[LossFlatCfg, LossFlatCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        #dump: dict,
    ) -> Float[Tensor, ""]:
        min_scale = torch.min(gaussians.scales.abs(),dim=-1)[0]
        # print('I AM HERE')
        # Compute average of minimum eigenvalues
        return self.cfg.weight * min_scale.mean()