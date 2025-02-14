import os

import numpy as np
import torch
from pathlib import Path
from .wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda")

class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """
    def __init__(self, model_det, model_pose, device='cuda'):
        self.args = model_det, model_pose, device

    def release_memory(self):
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc; gc.collect()

    def __call__(self, oriImg):
        if not hasattr(self, 'pose_estimation'):
            self.pose_estimation = Wholebody(*self.args)

        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose
        
current_dir = Path(__file__).resolve().parent

# For Docker container
pretrained_weights_dir = Path("/usr/app/pretrained_weights")
model_det_path = pretrained_weights_dir / "yolox_l.onnx"
model_pose_path = pretrained_weights_dir / "dw-ll_ucoco_384.onnx"

# For local usage 
# pointing to the pretrained_weights directory
# model_det_path = "/home/hertzai2019/AnimateX/echomimic_v2/pretrained_weights/yolox_l.onnx"
# model_pose_path = "/home/hertzai2019/AnimateX/echomimic_v2/pretrained_weights/dw-ll_ucoco_384.onnx"

dwpose_detector = DWposeDetector(
    model_det=str(model_det_path),
    model_pose=str(model_pose_path),
    device=device)
print('dwpose_detector init ok', device)
