import torch
import torch.nn as nn
import math
import itertools

import mentat_lss.models.blocks as blocks

class single_zbin_transformer(nn.Module):
    """Class defining a single independent transformer network"""

    def __init__(self, config_dict):
        """Initializes an individual network, responsible for outputting a portion of the full model vector. The user is not meant to call this function directly.

        Args:
            config_dict: input dictionary with various network architecture options.
        """
        super().__init__()

        # TODO: Allow specification of activation function        
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        self.input_dim = config_dict["num_cosmo_params"] + (2 * config_dict["num_nuisance_params"])
        self.output_dim = self.num_ells * self.num_kbins
        num_spectra = config_dict["num_tracers"] + math.comb(config_dict["num_tracers"], 2)

        # Fixed one-hot offset per spectrum type, added to input before the first linear layer.
        # Shape [num_spectra, input_dim]: row s has a 1 in position s, 0s elsewhere.
        spectrum_bias = torch.zeros(num_spectra, self.input_dim)
        spectrum_bias[:, :num_spectra] = torch.eye(num_spectra)
        self.register_buffer("spectrum_bias", spectrum_bias)
        
        # mlp blocks
        self.input_layer = nn.Linear(self.input_dim, self.output_dim)
        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["galaxy_ps_emulator"]["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(self.output_dim,
                                        self.output_dim,
                                        config_dict["galaxy_ps_emulator"]["num_block_layers"],
                                        config_dict["galaxy_ps_emulator"]["use_skip_connection"]))
        
        # expand mlp section output
        split_dim = config_dict["galaxy_ps_emulator"]["split_dim"]
        split_size = config_dict["galaxy_ps_emulator"]["split_size"]
        embedding_dim = split_size*split_dim
        self.embedding_layer = nn.Linear(self.output_dim, embedding_dim)

        # do one transformer block per z-bin for now
        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["galaxy_ps_emulator"]["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1))
            self.transformer_blocks.add_module("Activation"+str(i+1), 
                    blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)

    def forward(self, input_params:torch.Tensor, spectrum_indices:torch.Tensor=None):
        """Passes an input tensor through the network"""

        X = input_params
        if spectrum_indices is not None:
            X = X + self.spectrum_bias[spectrum_indices]
        X = self.input_layer(X)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_layer(X)

        return X

class combined_tracer_transformer(nn.Module):
    """Class defining a stack of single_zbin_transformer objects, one for each redshift bin of the power spectrum output"""

    def __init__(self, config_dict):
        """Initializes a group of single_zbin_transformer based on the input dictionary.
        
        This function creates nz*nps total networks, where nz is the number of redshift bins, and nps
        is the number of auto + cross power spectra per redshift bin.

        Args:
            config_dict: input dictionary with various network architecture options.  
        """
        super().__init__()

        # output dimensions
        self.num_zbins = config_dict["num_zbins"]
        self.num_spectra = config_dict["num_tracers"] +  math.comb(config_dict["num_tracers"], 2)
        self.num_tracers = config_dict["num_tracers"]
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.num_cosmo_params    = config_dict["num_cosmo_params"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        self.output_dim = self.num_ells * self.num_kbins

        # Stores networks sequentially in a list
        self.networks = nn.ModuleList()
        for z in range(self.num_zbins):
            self.networks.append(single_zbin_transformer(config_dict))

        # Precompute tracer index pairs in spectrum order: (0,0), (0,1), ..., (nt-1,nt-1)
        pairs = [(i1, i2) for i1, i2 in itertools.product(range(self.num_tracers), repeat=2) if i1 <= i2]
        self.register_buffer("_tracer_idx1", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("_tracer_idx2", torch.tensor([p[1] for p in pairs], dtype=torch.long))

    def organize_parameters(self, input_params):
        """Organizes input cosmology + bias parameters by redshift only and treats spectra
        (tracer pairs) as extra batch entries.

        Args:
            input_params: tensor with shape [batch, total_param_length]. `total_param_length`
                is expected to be `num_cosmo_params + num_nuisance_params*(num_zbins*num_tracers)`

        Returns:
            organized_params: tensor with shape [batch * num_spectra, num_zbins,
                num_cosmo_params + 2*num_nuisance_params]. Each output batch entry corresponds
                to a single spectrum (tracer pair) for one original batch sample, and the
                second dimension runs over redshift bins.
        """
        bsize = input_params.shape[0]

        # Cosmo params are the same for every spectrum and z-bin
        # [batch, nc] → [batch, num_spectra, num_zbins, nc]
        cosmo = input_params[:, :self.num_cosmo_params]
        cosmo = cosmo[:, None, None, :].expand(bsize, self.num_spectra, self.num_zbins, -1)

        # The nuisance block has layout [k, z, t] in C-order, so reshape directly.
        # [batch, nn*nz*nt] → [batch, nn, nz, nt] → [batch, nz, nt, nn]
        nuisance = input_params[:, self.num_cosmo_params:]
        nuisance = nuisance.reshape(bsize, self.num_nuisance_params, self.num_zbins, self.num_tracers)
        nuisance = nuisance.permute(0, 2, 3, 1)  # [batch, nz, nt, nn]

        # Index nuisance by tracer pair → [batch, nz, num_spectra, nn] → [batch, num_spectra, nz, nn]
        nus1 = nuisance[:, :, self._tracer_idx1, :].permute(0, 2, 1, 3)
        nus2 = nuisance[:, :, self._tracer_idx2, :].permute(0, 2, 1, 3)

        # Concatenate along param dim and flatten batch × spectrum → [batch*num_spectra, nz, nc+2*nn]
        organized_params = torch.cat([cosmo, nus1, nus2], dim=-1)
        return organized_params.reshape(bsize * self.num_spectra, self.num_zbins,
                                        self.num_cosmo_params + 2 * self.num_nuisance_params)

    def forward(self, input_params, net_idx = None):
        """Passes an input tensor through the network"""

        # input_params has shape [batch*num_spectra, nz, params]; spectrum indices repeat
        # as [0, 1, ..., num_spectra-1, 0, 1, ...] across the leading dimension.
        bsize_total = input_params.shape[0]
        spectrum_indices = torch.arange(self.num_spectra, device=input_params.device).repeat(
            bsize_total // self.num_spectra)

        # feed parameters through all sub-networks
        if net_idx == None:
            X = torch.zeros((bsize_total, self.num_zbins, self.output_dim), device=input_params.device)
            for z in range(self.num_zbins):
                X[:, z] = self.networks[z](input_params[:,z], spectrum_indices)

        # feed parameters through an individual sub-network (used in training)
        else:
            X = self.networks[net_idx](input_params[:,net_idx], spectrum_indices)

        return X
        