import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from utils import init_weights, init_weights_orthogonal_normal
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class AxisAlignedConvGaussianPrior(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers):
        super(AxisAlignedConvGaussianPrior, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)


    def forward(self, x):
        x = x.to(torch.float32)
        encoding = self.encoder(x)
        encoding = torch.mean(encoding, dim=(2, 3), keepdim=True)
        mu_log_sigma = self.conv_layer(encoding).squeeze(-1).squeeze(-1)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        return Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

class Fcomb(nn.Module):
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.use_tile = use_tile

        if self.use_tile:
            layers = [nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True)]
            for _ in range(no_convs_fcomb - 2):
                layers.extend([nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True)])
            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(256, 256, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        if self.use_tile:
            z = z.unsqueeze(2).unsqueeze(2)
            z = self.tile(z, 2, feature_map.size(2))
            z = self.tile(z, 3, feature_map.size(3))
            feature_map = torch.cat((feature_map, z), dim=1)
            output = self.layers(feature_map)
            return self.last_layer(output)

class P2SAM(nn.Module):
    def __init__(self, device, lora_ckpt, input_channels=1, num_classes=8, img_size=128, num_filters=[32, 64, 128, 192], latent_dim=256, no_convs_fcomb=4, beta=10.0):
        super(P2SAM, self).__init__()
        self.device = device
        self.img_size = img_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta

        self.sam, self.img_embedding_size = sam_model_registry["vit_b"](
            image_size=self.img_size,
            num_classes=self.num_classes,
            checkpoint="sam_vit_b_01ec64.pth",
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1]
        )
        self.sam.to(self.device)
        self.lora_sam = LoRA_Sam(self.sam, 4).to(self.device)
        self.lora_sam.load_lora_parameters(lora_ckpt)
        self.prior_dense = AxisAlignedConvGaussianPrior(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers).to(self.device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)

    def forward(self, batch_input, batch_input_ori, input_size=128, train=True):
        img_size = input_size
        self.prior_dense_latent_space = self.prior_dense(batch_input_ori)
        input_images = self.lora_sam.sam.preprocess(batch_input)
        image_embeddings = self.lora_sam.sam.image_encoder(input_images)
        sparse_embeddings, dense_embeddings = self.lora_sam.sam.prompt_encoder(points=None, boxes=None, masks=None)
        batch_shape = batch_input.size(0)
        dense_embeddings = dense_embeddings.repeat(batch_shape, 1, 1, 1)

        if train:
            z_posterior_dense = self.prior_dense_latent_space.rsample()
        else:
            z_posterior_dense = self.prior_dense_latent_space.sample()

        dense_embeddings_ditsturb = self.fcomb(dense_embeddings, z_posterior_dense)
        low_res_masks, iou_predictions = self.lora_sam.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.lora_sam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings_ditsturb,
            multimask_output=True
        )
        masks = self.lora_sam.sam.postprocess_masks(
            low_res_masks,
            input_size=(img_size, img_size),
            original_size=(128, 128)
        )

        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }