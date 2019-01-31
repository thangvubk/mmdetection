import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, gt_bboxes=None, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.gt_bboxes = gt_bboxes

    def add_gt_(self, gt_labels, gt_self_inds=None):
        if gt_self_inds is not None:
            self_inds = gt_self_inds
        else:
            self_inds = torch.arange(
                1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
