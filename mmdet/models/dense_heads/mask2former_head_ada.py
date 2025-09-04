from typing import List, Tuple
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor
from ..utils import get_uncertain_point_coords_with_randomness
from mmdet.utils import reduce_mean
from mmcv.ops import point_sample
from mmdet.registry import MODELS
from .mask2former_head import Mask2FormerHead

@MODELS.register_module()
class Mask2FormerHeadADA(Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 loss_local_consistent=None,
                 loss_negative_learning=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Build extra losses if configs are provided
        self.negative_learning_loss = None
        self.local_consistent_loss = None
        if loss_local_consistent is not None:
            self.local_consistent_loss = MODELS.build(loss_local_consistent)
        if loss_negative_learning is not None:
            self.negative_learning_loss = MODELS.build(loss_negative_learning)

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (_, _, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if 'syn' in batch_img_metas[0]['seg_map_path']:
            domain = 'source'
        elif 'phenobench' in batch_img_metas[0]['seg_map_path']:
            domain = 'target'
        mask_targets = torch.cat(mask_targets_list, dim=0)  # [C x H x W]
        mask_weights = torch.stack(mask_weights_list, dim=0)# [B x Q] where Q is num_queries e.g. 100
        mask_preds = F.interpolate(                         # [B x Q x 128 x 128] --> [B x Q x H x W]
            mask_preds, 
            size=mask_targets.shape[-2:], 
            mode='bilinear', 
            align_corners=False)
        mask_preds = mask_preds[mask_weights > 0]           # [B x Q x H x W] --> [B x 3 x H x W]
        mask_preds_prob = torch.softmax(mask_preds, dim=1)
        # Add extra losses if enabled
        extra_losses = {}
        if mask_targets.shape[0] == 0:
            # zero match
            # If the labels are set to 255 then it should yield 0 loss value. 
            loss_dice = mask_preds.sum() * 0.0 
            loss_mask = mask_preds.sum() * 0.0
            if self.local_consistent_loss is not None:
                extra_losses['loss_local_consistent'] = mask_preds.sum() * 0.0
            if self.negative_learning_loss is not None:
                extra_losses['loss_negative'] = mask_preds.sum() * 0.0
            return loss_cls, loss_mask, loss_dice, extra_losses
        # Source domain only
        if domain == 'source':
            if self.local_consistent_loss is not None:
                extra_losses['loss_local_consistent'] = self.local_consistent_loss(mask_preds.unsqueeze(1), mask_targets)
            if self.negative_learning_loss is not None:
                extra_losses['loss_negative'] = mask_preds.sum() * 0.0
        # Target domain only 
        if domain == 'target':
            if self.local_consistent_loss is not None:
                extra_losses['loss_local_consistent'] = mask_preds.sum() * 0.0
            if self.negative_learning_loss is not None:
                extra_losses['loss_negative'] = self.negative_learning_loss(mask_preds_prob)
        return loss_cls, loss_mask, loss_dice, extra_losses