from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    depth: Float[Tensor, "batch view 1 h w"]
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian 4 4"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 2"]
    rotations: Float[Tensor, "batch gaussian 4"]
