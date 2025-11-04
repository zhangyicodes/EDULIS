# Copyright (c) OpenMMLab. All rights reserved.
from .bfp import BFP
from .channel_mapper import ChannelMapper
from .cspnext_pafpn import CSPNeXtPAFPN
from .ct_resnet_neck import CTResNetNeck
from .dilated_encoder import DilatedEncoder
from .dyhead import DyHead
from .fpg import FPG
from .fpn import FPN
from .fpn_unfolding import FPN_UNF
from .fpn_carafe import FPN_CARAFE
from .fpn_dropblock import FPN_DropBlock
from .hrfpn import HRFPN
from .nas_fpn import NASFPN
from .nasfcos_fpn import NASFCOS_FPN
from .pafpn import PAFPN
from .rfp import RFP
from .ssd_neck import SSDNeck
from .ssh import SSH
from .yolo_neck import YOLOV3Neck
from .yolox_pafpn import YOLOXPAFPN

__all__ = [
    'FPN', 'FPN_UNF','BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
    'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN', 'DyHead', 'CSPNeXtPAFPN', 'SSH',
    'FPN_DropBlock'
]

# fpn_unf_instance = FPN_UNF(in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_output',  # use P5
#         num_outs=5,
#         relu_before_extra_convs=True)
# for name, param in fpn_unf_instance.named_parameters():
#     if not param.requires_grad:
#         print(f"Parameter {name} does not require gradient.")
