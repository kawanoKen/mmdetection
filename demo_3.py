from mmdet.uotod.src.uotod.match import BalancedSinkhorn
from mmdet.uotod.src.uotod.loss import GIoULoss
import torch

ot = BalancedSinkhorn(
    loc_match_module=GIoULoss(reduction="none"),
    background_cost=0.,
)

cost = torch.ones(3, 5)
a = torch.tensor([
    [2, 3, 4],
    [2, 1]
])
breakpoint()