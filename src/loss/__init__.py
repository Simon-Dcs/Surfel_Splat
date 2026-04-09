from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_flat import LossFlat, LossFlatCfgWrapper
from .loss_geo import LossGeo, LossGeoCfgWrapper
from .loss_normal import LossNormal, LossNormalCfgWrapper
from .loss_dist import LossDist, LossDistCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossFlatCfgWrapper: LossFlat,
    LossGeoCfgWrapper: LossGeo,
    LossNormalCfgWrapper: LossNormal,
    LossDistCfgWrapper: LossDist,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossFlatCfgWrapper | LossGeoCfgWrapper | LossNormalCfgWrapper | LossDistCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
