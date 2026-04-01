import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import itertools

from mentat_lss.models.blocks import block_cov_resnet
from mentat_lss.utils import rearange_to_full


class cov_network(nn.Module):
    """Neural network for emulating covariance matrices for a single redshift bin.

    Supports two architectures:
    - "MLP": a series of residual MLP blocks mapping cosmology + bias parameters to
      the compressed lower-triangular Cholesky factor of a covariance matrix.
    - "MLP-T": an MLP backbone followed by transformer encoder blocks that
      refine the output by reasoning over spatially-adjacent matrix patches.

    Note: this class expects pre-normalized inputs. Normalization is handled
    externally by cov_emulator, consistent with ps_emulator conventions.
    """

    def __init__(self, config_dict:dict):
        """Initializes the network based on the input configuration dictionary.

        Args:
            config_dict (dict): network configuration. Expected keys:
                architecture (str): one of ["MLP", "MLP-T"]
                input_dim (int): number of input parameters per network
                    (num_cosmo_params + num_nuisance_params * num_tracers)
                output_dim (int): dimension N of the output N×N covariance matrix
                mlp_dims (list[int]): list of MLP layer widths, length = num_mlp_blocks + 1
                num_mlp_blocks (int): number of residual MLP blocks
                num_transformer_blocks (int): number of transformer encoder layers (MLP-T only)
                num_heads (int): number of attention heads (MLP-T only)
                patch_size (list[int]): [h, w] patch size for patchification (MLP-T only)
                embedding (bool): whether to add sinusoidal positional embeddings (MLP-T only)
                dropout_prob (float): dropout probability in transformer blocks (MLP-T only)

        Raises:
            KeyError: if architecture is not one of ["MLP", "MLP-T"]
        """
        super().__init__()

        self.architecture = config_dict["architecture"]
        self.input_dim  = config_dict["input_dim"]
        self.output_dim = config_dict["output_dim"]

        # compressed matrix shape: (output_dim+1, output_dim/2)
        self.N      = torch.Tensor([config_dict["output_dim"] + 1, config_dict["output_dim"] / 2]).int()
        self.N_flat = (self.N[0] * self.N[1]).item()

        # ---------------------------------------------------------------
        # MLP architecture
        if self.architecture == "MLP":
            self.h1 = nn.Linear(self.input_dim, config_dict["mlp_dims"][0])
            self.mlp_blocks = nn.Sequential()
            for i in range(config_dict["num_mlp_blocks"]):
                self.mlp_blocks.add_module(
                    "ResNet" + str(i + 1),
                    block_cov_resnet(config_dict["mlp_dims"][i], config_dict["mlp_dims"][i + 1]))
            self.out = nn.Linear(config_dict["mlp_dims"][-1], self.N_flat)

        # ---------------------------------------------------------------
        # MLP + Transformer architecture
        elif self.architecture == "MLP-T":
            self.h1 = nn.Linear(self.input_dim, config_dict["mlp_dims"][0])
            self.mlp_blocks = nn.Sequential()
            for i in range(config_dict["num_mlp_blocks"]):
                self.mlp_blocks.add_module(
                    "ResNet" + str(i + 1),
                    block_cov_resnet(config_dict["mlp_dims"][i], config_dict["mlp_dims"][i + 1]))
            self.out = nn.Linear(config_dict["mlp_dims"][-1], self.N_flat)

            self.patch_size  = torch.Tensor(config_dict["patch_size"]).int().tolist()
            self.n_patches   = [(self.N[0].item() // self.patch_size[0]),
                                 (self.N[1].item() // self.patch_size[1])]
            sequence_len     = int(self.patch_size[0] * self.patch_size[1])
            num_sequences    = self.n_patches[0] * self.n_patches[1]
            self.embedding   = config_dict["embedding"]

            self.linear_map = nn.Linear(sequence_len, sequence_len)
            self.transform_blocks = nn.Sequential()
            for i in range(config_dict["num_transformer_blocks"]):
                self.transform_blocks.add_module(
                    "transform" + str(i + 1),
                    nn.TransformerEncoderLayer(sequence_len, config_dict["num_heads"],
                                               4 * sequence_len, config_dict["dropout_prob"],
                                               "gelu", batch_first=True))

            pos_embed = self._get_positional_embedding(num_sequences, sequence_len)
            pos_embed.requires_grad = False
            self.register_buffer("pos_embed", pos_embed)

        else:
            raise KeyError(f"Invalid architecture! Must be one of ['MLP', 'MLP-T'], but got {self.architecture}")

    def load_pretrained(self, path:str, freeze:bool=True):
        """Loads pre-trained layer weights from file into the current model.

        Args:
            path (str): path to the saved network state dict file
            freeze (bool, optional): if True, freezes loaded weights. Defaults to True.
        """
        pre_trained_dict = torch.load(path, weights_only=True)
        for name, param in pre_trained_dict.items():
            if name not in self.state_dict():
                continue
            self.state_dict()[name].copy_(param)
            if freeze:
                self.state_dict()[name].requires_grad = False

    def _get_positional_embedding(self, sequence_length:int, d:int):
        """Constructs a sinusoidal positional embedding.

        Args:
            sequence_length (int): number of independent sequences (patches)
            d (int): size of each sequence (flattened patch dimension)

        Returns:
            embeddings (torch.Tensor): positional embedding with shape (sequence_length, d)
        """
        embeddings = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    embeddings[i][j] = np.sin(i / (10000 ** (j / d)))
                else:
                    embeddings[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))
        return embeddings

    def _patchify(self, X:torch.Tensor):
        """Splits the compressed matrix into spatially-adjacent patches.

        Args:
            X (torch.Tensor): compressed matrix from the MLP, shape (batch_size, N_flat)

        Returns:
            patches (torch.Tensor): shape (batch_size, n_patches_h * n_patches_w, patch_h * patch_w)
        """
        X = X.reshape(-1, self.N[0].item(), self.N[1].item())
        patches = X.unfold(1, self.patch_size[0], self.patch_size[0])\
                   .unfold(2, self.patch_size[1], self.patch_size[1])
        patches = patches.reshape(-1, self.n_patches[1] * self.n_patches[0],
                                  self.patch_size[0] * self.patch_size[1])
        return patches

    def _un_patchify(self, patches:torch.Tensor):
        """Recombines patches into a full compressed matrix.

        Args:
            patches (torch.Tensor): shape (batch_size, n_patches_h * n_patches_w, patch_h * patch_w)

        Returns:
            X (torch.Tensor): reconstructed matrix, shape (batch_size, N[0], N[1])
        """
        patches = patches.reshape(-1, self.n_patches[0], self.n_patches[1],
                                  self.patch_size[0], self.patch_size[1])
        X = patches.permute(0, 1, 3, 2, 4).contiguous()
        X = X.view(-1, self.N[0].item(), self.N[1].item())
        return X

    def forward(self, X:torch.Tensor):
        """Runs the forward pass from (pre-normalized) parameters to Cholesky factor.

        Args:
            X (torch.Tensor): batch of pre-normalized input parameters,
                shape (batch_size, input_dim)

        Returns:
            X (torch.Tensor): batch of lower-triangular matrices,
                shape (batch_size, output_dim, output_dim)
        """
        if self.architecture == "MLP":
            X = F.leaky_relu(self.h1(X))
            for blk in self.mlp_blocks:
                X = F.leaky_relu(blk(X))
            X = torch.tanh(self.out(X))
            X = X.view(-1, self.N[0].item(), self.N[1].item())
            X = rearange_to_full(X, lower_triangular=True)
            return X

        elif self.architecture == "MLP-T":
            X = F.leaky_relu(self.h1(X))
            for blk in self.mlp_blocks:
                X = F.leaky_relu(blk(X))
            Y = torch.tanh(self.out(X))

            X = self.linear_map(self._patchify(Y))
            if self.embedding:
                X = X + self.pos_embed.repeat(Y.shape[0], 1, 1)
            for blk in self.transform_blocks:
                X = blk(X)
            X = torch.tanh(self._un_patchify(X)) + Y.view(-1, self.N[0].item(), self.N[1].item())
            X = X.view(-1, self.N[0].item(), self.N[1].item())
            X = rearange_to_full(X, lower_triangular=True)
            return X


class stacked_cov_network(nn.Module):
    """Stack of cov_network objects, one per redshift bin.

    Analogous to stacked_mlp, this class manages num_zbins independent
    cov_network instances and provides parameter organization to route the
    flat cosmology + bias parameter vector to each per-zbin network.
    """

    def __init__(self, config_dict:dict):
        """Initializes one cov_network per redshift bin.

        The per-network input dimension is derived from the config:
            input_dim = num_cosmo_params + num_nuisance_params * num_tracers

        Args:
            config_dict (dict): network configuration. In addition to the keys
                required by cov_network, expects:
                num_zbins (int): number of independent redshift bins
                num_tracers (int): number of galaxy tracers
                num_cosmo_params (int): number of cosmological parameters
                num_nuisance_params (int): number of nuisance parameters per tracer
        """
        super().__init__()

        self.num_zbins          = config_dict["num_zbins"]
        self.num_tracers        = config_dict["num_tracers"]
        self.num_cosmo_params   = config_dict["num_cosmo_params"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        # each per-zbin network receives cosmo params + all tracers' bias params
        per_net_input_dim = self.num_cosmo_params + self.num_nuisance_params * self.num_tracers
        net_config = dict(config_dict)
        net_config["input_dim"] = per_net_input_dim

        self.networks = nn.ModuleList(
            [cov_network(net_config) for _ in range(self.num_zbins)])

    def organize_parameters(self, flat_params:torch.Tensor):
        """Organizes a flat parameter vector into per-zbin inputs.

        Follows the same tracer/zbin ordering convention as stacked_mlp:
        within the bias portion of flat_params, the stride is
        num_tracers * num_zbins, so parameter type varies slowest.

        Args:
            flat_params (torch.Tensor): input parameters with shape
                (batch, num_cosmo_params + num_nuisance_params * num_tracers * num_zbins)

        Returns:
            org_params (torch.Tensor): per-zbin parameters with shape
                (batch, num_zbins, num_cosmo_params + num_nuisance_params * num_tracers)
        """
        batch = flat_params.shape[0]
        per_net_dim = self.num_cosmo_params + self.num_nuisance_params * self.num_tracers
        org_params = torch.zeros(batch, self.num_zbins, per_net_dim, device=flat_params.device)

        # cosmology parameters are shared across all zbins
        org_params[:, :, :self.num_cosmo_params] = \
            flat_params[:, :self.num_cosmo_params].unsqueeze(1)

        # bias parameters: stride = num_tracers * num_zbins (same as stacked_mlp)
        stride = self.num_tracers * self.num_zbins
        for z in range(self.num_zbins):
            bias_start = self.num_cosmo_params
            for t in range(self.num_tracers):
                src_idx = (z * self.num_tracers) + t
                dst_start = bias_start + t * self.num_nuisance_params
                dst_end   = dst_start + self.num_nuisance_params
                org_params[:, z, dst_start:dst_end] = \
                    flat_params[:, self.num_cosmo_params + src_idx::stride]

        return org_params

    def forward(self, org_params:torch.Tensor, z_idx:int=None):
        """Runs pre-normalized per-zbin parameters through the network(s).

        Args:
            org_params (torch.Tensor): pre-normalized parameters, shape
                (batch, num_zbins, per_net_input_dim)
            z_idx (int, optional): if given, runs only the network for that
                redshift bin and returns shape (batch, output_dim, output_dim).
                If None, runs all zbins and returns shape
                (batch, num_zbins, output_dim, output_dim). Defaults to None.

        Returns:
            X (torch.Tensor): covariance Cholesky factors (see z_idx description)
        """
        if z_idx is not None:
            return self.networks[z_idx](org_params[:, z_idx])

        output_dim = self.networks[0].output_dim
        X = torch.zeros(org_params.shape[0], self.num_zbins, output_dim, output_dim,
                        device=org_params.device)
        for z in range(self.num_zbins):
            X[:, z] = self.networks[z](org_params[:, z])
        return X
