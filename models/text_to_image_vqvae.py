import torch
import torch.nn as nn
import numpy as np
from models.vqvae import VQVAE
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer
from transformers import AutoTokenizer, CLIPModel, AutoProcessor

class Grounded_VQVAE(VQVAE):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 vqvae_dict, clip_config):
        # Initialize the original VQVAE module
        super().__init__(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta)
        self.num_channels = 3
        self.res_h_dim    = res_h_dim
        # Initialize CLIP Encoder and tokenizer
        self.clip_encoder = CLIPModel(clip_config)
        self.tokenizer    = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize Linear Layer Stack
        self.grounding_stack = = nn.ModuleList([
                    nn.Linear(clip_config.projection_dim, res_h_dim ** 2),
                    nn.Linear(clip_config.projection_dim, res_h_dim ** 2),
                    nn.Linear(clip_config.projection_dim, res_h_dim ** 2)
                ])
        """
        TODO: Load in pre-trained VQVAE AND Freeze CLIP Modules
        """

    def _load_from_pretrained(self, codebook_dict, decoder_dict):
        """
        TODO: Fix Later! This needs to only load in the VQVAE
        """
        pass

    def _freeze_modules(self):
        """
        TODO: freeze CLIP modules
        """
        pass

    def encode_input(x):
        # Encode our text
        inputs = self.tokenizer(x, padding=True, return_tensors="pt")
        text_features = self.clip_encoder.get_text_features(**inputs)
        # Get the batch size
        b, _ = text_features.shape
        # Project it into the encoder input dimension
        grounded_features = torch.stack([layer(text_features) for layer in self.grounding_stack],dim = 1)
        # Return a [b, 3, res_h_dim, res_h_dim] view of our grounded text features
        return grounded_features.view((b,self.num_channels,res_dim_sqrt,res_dim_sqrt))

    def forward(self, x):
        # Assuming x is already projected into the VQVAE encoder dim, do a forward pass of VQVAE
        return super().forward(x)


class Fusion_VQVAE(Grounded_VQVAE):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 codebook_dict, decoder_dict, clip_config, fusion_fn):
        # Initialize a Grounded_VQVAE
        super().__init__(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, None, clip_config)
        # Initialize CLIP Image Processor
        self.processor    = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize fusion module; Type will be (x,y) -> z
        self.fusion_module = fusion_fn
        """
        TODO: Load in pre-trained VQVAE AND Freeze CLIP Modules
        """

    def encode_fused_input(self, x, y):
        # Encode our text
        inputs = self.tokenizer(x, padding=True, return_tensors="pt")
        text_features = self.clip_encoder.get_text_features(**inputs)
        # Encode our image
        img_in = self.processor(images = y, return_tensors="pt")
        img_features = self.clip_encoder.get_image_features(**img_in)
        # Fuse features together; Fusion Module preserves Input Shape here
        fused_features = self.fusion_module(text_features, img_features)
        # Get the batch size
        b, _ = text_features.shape
        # Project it into the encoder input dimension
        grounded_features = torch.stack([layer(fused_features) for layer in self.grounding_stack],dim = 1)
        # Return a [b, 3, res_h_dim, res_h_dim] view of our grounded text features
        return grounded_features.view((b,self.num_channels,res_dim_sqrt,res_dim_sqrt))

    def forward(self, x):
        # Assuming input is fused and/or grounded, pass into the VQVAE
        return super().forward(x)

class LoRA_Grounded_VQVAE(Grounded_VQVAE):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 codebook_dict, decoder_dict, clip_config, fusion_fn):
        # Initialize a Grounded_VQVAE
        super().__init__(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, None, clip_config)
        # Initialize CLIP Image Processor
        self.processor    = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        """
        TODO: Load in pre-trained VQVAE AND Freeze CLIP Modules
        """
