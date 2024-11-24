import torch
import torch.nn as nn
import numpy as np
from models.vqvae import VQVAE
from models.quantizer import VectorQuantizer
from transformers import AutoTokenizer, CLIPModel, AutoProcessor

class Grounded_VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 codebook_dict, decoder_dict, clip_config):
        super(Grounded_VQVAE, self).__init__()
        # Initialize the codebook and decoder
        self.codebook = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder  = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        # Initialize CLIP Encoder and tokenizer
        self.clip_encoder = CLIPModel(clip_config)
        self.tokenizer    = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize Linear Layer
        self.grounding_layer = nn.Linear(clip_config.projection_dim, embedding_dim)

    def _load_from_pretrained(self, codebook_dict, decoder_dict):
        # Load in the pre-trained decoder and codebook
        self.codebook.load_state_dict(codebook_dict)
        self.decoder.load_state_dict(decoder_dict)

    def forward(self, x):
        # Encode our text
        inputs = self.tokenizer(x, padding=True, return_tensors="pt")
        text_features = self.clip_encoder.get_text_features(**inputs)
        # Project it into the codebook input dimension
        proj_txt_features = self.grounding_layer(text_features)
        # Quantize our text features
        embedding_loss, z_q, perplexity, _, _ = self.codebook(proj_txt_features)
        # Decode our codebook text vector
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity


class Fusion_VQVAE(Grounded_VQVAE):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 codebook_dict, decoder_dict, clip_config, fusion_fn):
        # Initialize a Grounded_VQVAE, since they have many similar components
        super(Fusion_VQVAE, self).__init__()
        # Initialize CLIP Image Processor
        self.processor    = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Initialize fusion module; Type will be (x,y) -> z
        self.fusion_module = fusion_fn
        # Load in pretrained codebook and deocder
        self._load_from_pretrained(codebook_dict, decoder_dict)

    def forward(self, x, y):
        # x -> Text, y -> image
        if self.training and y is not None:
            # Encode our text
            text_in = self.tokenizer(x, padding=True, return_tensors="pt")
            text_features = self.clip_encoder.get_text_features(**text_in)
            # Encode our image
            img_in = self.processor(images = y, return_tensors="pt")
            img_features = self.clip_encoder.get_image_features(**img_in)
            # Fuse features together
            fused_features = self.fusion_module(text_features, img_features)
            # Project features into codebook dimension
            proj_txt_features = self.grounding_layer(text_features)
            # Quantize our text features
            embedding_loss, z_q, perplexity, _, _ = self.codebook(proj_txt_features)
            # Decode our codebook text vector
            x_hat = self.decoder(z_q)
            return embedding_loss, x_hat, perplexity
        else:
            # This is the same as our Grounded_VQVAE forward function, so we just call that here
            return super().forward(x)
