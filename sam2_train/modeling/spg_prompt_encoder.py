import torch
from torch import nn
from sam2_train.modeling.sam.prompt_encoder import PromptEncoder

class SPGPromptEncoder(nn.Module):
    """Self Prompt Generator using image mirroring differences."""

    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans=16):
        super().__init__()
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
        )

    def forward(self, image, mirrored_image):
        """Generate prompt embeddings from image and its mirrored counterpart."""
        diff = (image - mirrored_image).abs().mean(1, keepdim=True)
        thresh = diff.mean(dim=(2, 3), keepdim=True) + diff.std(dim=(2, 3), keepdim=True)
        mask = (diff > thresh).float()
        B, _, H, W = mask.shape
        coords = torch.zeros(B, 1, 2, device=image.device)
        labels = torch.ones(B, 1, dtype=torch.int32, device=image.device)
        for i in range(B):
            loc = torch.nonzero(mask[i, 0], as_tuple=False)
            if loc.numel() > 0:
                center = loc.float().mean(0)
                coords[i, 0, 0] = center[1]
                coords[i, 0, 1] = center[0]
            else:
                coords[i, 0] = torch.tensor([W / 2, H / 2], device=image.device)
        sparse, dense = self.prompt_encoder(points=(coords, labels), boxes=None, masks=mask)
        return sparse, dense

    def get_dense_pe(self):
        return self.prompt_encoder.get_dense_pe()
