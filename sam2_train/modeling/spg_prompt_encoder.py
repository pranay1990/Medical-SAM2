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

    @property
    def mask_input_size(self):
        """Size of masks expected by the underlying prompt encoder."""
        return self.prompt_encoder.mask_input_size

    def forward(
        self,
        image=None,
        mirrored_image=None,
        *,
        points=None,
        boxes=None,
        masks=None,
        batch_size=-1,
    ):
        """Generate prompt embeddings.

        If ``image`` and ``mirrored_image`` are provided, embeddings are
        produced based on the difference between the images (self‑prompt
        generation). Otherwise this method proxies to the underlying
        ``PromptEncoder`` so it can be used transparently anywhere a regular
        ``PromptEncoder`` would be expected.
        """

        if image is not None and mirrored_image is not None:
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
            return self.prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=mask,
                batch_size=batch_size,
            )

        return self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
            batch_size=batch_size,
        )

    def get_dense_pe(self):
        return self.prompt_encoder.get_dense_pe()
