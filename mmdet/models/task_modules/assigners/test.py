import torch
import torch.nn.functional as F
from torch import Tensor

# gt_labels = torch.tensor([0, 1])
# pred_scores = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.1, 0.2, 0.4]])
# num_valid = 3

# gt_onehot_label = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().unsqueeze(0).repeat(num_valid, 1, 1)
# print(gt_onehot_label)
# valid_pred_scores
decoded_bboxes = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.1, 0.2, 0.4]])
# valid_mask = [False, True, False]
# valid_decoded_bbox = decoded_bboxes[valid_mask]
# print(valid_decoded_bbox)

# assigned_gt_inds = torch.tensor([1, 2, 3])
# matched_gt_inds = torch.tensor([6])
# assigned_gt_inds[valid_mask] = matched_gt_inds + 1
# print(assigned_gt_inds)