# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Tuple
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
import gc
from typing import Dict

INF = 1e8


def print_tensor_sizes():
    print("\n=== Current Tensor Memory Usage ===")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
    print("=====================================\n")



@MODELS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                          f'be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = MODELS.build(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._iter_count = 0  # 添加迭代计数器

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different
            resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None,
                 **kwargs) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``bbox_head`` forward.
        """
        outs = ()
        # backbone
        x = self.extract_feat(batch_inputs)
        # bbox_head
        positive_infos = None
        if self.with_bbox:
            assert batch_data_samples is not None
            bbox_outs = self.bbox_head.forward(x)
            outs = outs + (bbox_outs, )
            # It is necessary to use `bbox_head.loss` to update
            # `_raw_positive_infos` which will be used in `get_positive_infos`
            # positive_infos will be used in the following mask head.
            _ = self.bbox_head.loss(x, batch_data_samples, **kwargs)
            positive_infos = self.bbox_head.get_positive_infos()
        # mask_head
        if positive_infos is None:
            mask_outs = self.mask_head.forward(x)
        else:
            mask_outs = self.mask_head.forward(x, positive_infos)
        outs = outs + (mask_outs, )
        return outs



    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = dict()

        positive_infos = None
        # CondInst and YOLACT have bbox_head
        if self.with_bbox:
            bbox_losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

            # for name, param in self.neck.named_parameters():
            #     if param.grad is None:
            #         print(f"neck Parameter {name} has no gradient.")

            losses.update(bbox_losses)
            # get positive information from bbox head, which will be used
            # in the following mask head.
            positive_infos = self.bbox_head.get_positive_infos()

        mask_loss = self.mask_head.loss(
            x, batch_data_samples, positive_infos=positive_infos, **kwargs)

        # 3. Mask branch
        # print("\n=== Before mask head ===")
        # with torch.cuda.amp.autocast():
        #     mask_loss = self.mask_head.loss(
        #         x, batch_data_samples, positive_infos=positive_infos, **kwargs)

        # print("\n=== After mask head ===")
        # print_tensor_sizes()  # 监控mask处理后的张量

        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        # for name, param in self.mask_head.named_parameters():
        #     if param.grad is None:
        #         print(f"Mask Parameter {name} has no gradient.")

        losses.update(mask_loss)
        # torch.cuda.empty_cache()

        self._iter_count += 1
        if self._iter_count % 5000 == 0:
            torch.cuda.empty_cache()

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Perform forward propagation of the mask head and predict mask
        results on the features of the upstream network.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
            - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = self.extract_feat(batch_inputs)
        if self.with_bbox:
            # the bbox branch does not need to be scaled to the original
            # image scale, because the mask branch will scale both bbox
            # and mask at the same time.
            bbox_rescale = rescale if not self.with_mask else False
            results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=bbox_rescale)
        else:
            results_list = None

        results_list = self.mask_head.predict(
            x, batch_data_samples, rescale=rescale, results_list=results_list)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
