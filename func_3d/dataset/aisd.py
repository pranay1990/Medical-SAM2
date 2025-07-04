import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from func_3d.utils import random_click, generate_bbox

class AISD(Dataset):
    """Dataset for Acute Ischemic Stroke Lesion Segmentation."""

    def __init__(
        self,
        args,
        data_path,
        case_ids=None,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt=None,
        seed=None,
        variation=0,
    ):
        self.image_root = os.path.join(data_path, "images")
        self.mask_root = os.path.join(data_path, "masks")
        if case_ids is None:
            case_ids = sorted(os.listdir(self.image_root))
        self.case_ids = case_ids
        self.img_size = args.image_size
        self.prompt = prompt
        self.transform = transform
        self.transform_msk = transform_msk
        self.mode = mode
        self.seed = seed
        self.variation = variation
        if mode == "Training":
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        img_dir = os.path.join(self.image_root, case_id)
        mask_dir = os.path.join(self.mask_root, case_id)
        frame_names = sorted(os.listdir(img_dir))
        num_frame = len(frame_names)
        if self.video_length is None:
            video_length = num_frame
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == "Training":
            start = np.random.randint(0, num_frame - video_length + 1)
        else:
            start = 0

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        bbox_dict = {}
        pt_dict = {}
        point_label_dict = {}

        newsize = (self.img_size, self.img_size)

        for i in range(video_length):
            name = frame_names[start + i]
            img = Image.open(os.path.join(img_dir, name)).convert("L")
            img = img.resize(newsize)
            img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)
            img_tensor[i] = img.repeat(3, 1, 1)

            mask_path = os.path.join(mask_dir, name)
            diff_obj_mask_dict = {}
            if os.path.exists(mask_path):
                msk = Image.open(mask_path).convert("L")
                msk = msk.resize(newsize)
                mask = torch.tensor(np.array(msk) > 0, dtype=torch.int32).unsqueeze(0)
                diff_obj_mask_dict[1] = mask
                if self.prompt == "bbox":
                    bbox = generate_bbox(
                        np.array(mask.squeeze(0)), variation=self.variation, seed=self.seed
                    )
                    bbox_dict[i] = {1: torch.tensor(bbox)}
                elif self.prompt == "click":
                    lbl, pt = random_click(np.array(mask.squeeze(0)), seed=self.seed)
                    pt_dict[i] = {1: torch.tensor(pt)}
                    point_label_dict[i] = {1: torch.tensor(lbl)}
            mask_dict[i] = diff_obj_mask_dict

        image_meta_dict = {"filename_or_obj": case_id}
        if self.prompt == "bbox":
            return {
                "image": img_tensor,
                "label": mask_dict,
                "bbox": bbox_dict,
                "image_meta_dict": image_meta_dict,
            }
        elif self.prompt == "click":
            return {
                "image": img_tensor,
                "label": mask_dict,
                "pt": pt_dict,
                "p_label": point_label_dict,
                "image_meta_dict": image_meta_dict,
            }
        else:
          return {"image": img_tensor, "label": mask_dict, "image_meta_dict": image_meta_dict}
