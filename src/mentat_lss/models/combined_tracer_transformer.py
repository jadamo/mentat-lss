import torch
import torch.nn as nn
import math
import itertools

import mentat_lss.models.blocks as blocks

class single_zbin_transformer(nn.Module):
    """Class defining a single independent transformer network"""

    def __init__(self, config_dict, is_cross_spectra:bool):
        """Initializes an individual network, responsible for outputting a portion of the full model vector. The user is not meant to call this function directly.

        Args:
            config_dict: input dictionary with various network architecture options.
            is_cross_spectra: specifies whether the network is responsible for cross or auto power spectra.
                Each case has a different input parameter size.
        """
        super().__init__()

        # TODO: Allow specification of activation function
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        num_tracers = config_dict["num_tracers"]
        num_spectra_this_net = num_tracers if not is_cross_spectra else math.comb(num_tracers, 2)
        spectrum_embed_dim = config_dict["galaxy_ps_emulator"].get("spectrum_embed_dim", 8)

        # Learned spectrum-type embedding, concatenated to the physical parameters.
        self.spectrum_embedding = nn.Embedding(num_spectra_this_net, spectrum_embed_dim)

        nuisance_blocks = 2 if is_cross_spectra else 1
        self.input_dim = config_dict["num_cosmo_params"] + (nuisance_blocks * config_dict["num_nuisance_params"]) + spectrum_embed_dim
        self.output_dim = self.num_ells * self.num_kbins

        # mlp blocks
        self.input_layer = nn.Linear(self.input_dim, self.output_dim)
        self.mlp_blocks = nn.Sequential()
        for i in range(config_dict["galaxy_ps_emulator"]["num_mlp_blocks"]):
            self.mlp_blocks.add_module("ResNet"+str(i+1),
                    blocks.block_resnet(self.output_dim,
                                        self.output_dim,
                                        config_dict["galaxy_ps_emulator"].get("hidden_dim_factor", 1.0),
                                        config_dict["galaxy_ps_emulator"]["num_block_layers"],
                                        "layer",
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
            X = torch.cat([X, self.spectrum_embedding(spectrum_indices)], dim=-1)
        X = self.input_layer(X)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_layer(X)

        return X

class combined_tracer_transformer(nn.Module):
    """Class defining a stack of single_zbin_transformer objects, one for each redshift bin of the power spectrum output.

    Each z-bin uses two networks: one for auto spectra and one for cross spectra.
    """

    def __init__(self, config_dict):
        """Initializes auto and cross networks for each z-bin based on the input dictionary.

        Args:
            config_dict: input dictionary with various network architecture options.
        """
        super().__init__()

        # output dimensions
        self.num_zbins = config_dict["num_zbins"]
        self.num_tracers = config_dict["num_tracers"]
        self.num_spectra = self.num_tracers + math.comb(self.num_tracers, 2)
        self.num_auto_spectra  = self.num_tracers
        self.num_cross_spectra = math.comb(self.num_tracers, 2)
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]

        self.num_cosmo_params    = config_dict["num_cosmo_params"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        self.output_dim = self.num_ells * self.num_kbins

        # One auto network and one cross network per z-bin
        self.auto_networks  = nn.ModuleList([single_zbin_transformer(config_dict, is_cross_spectra=False)
                                             for _ in range(self.num_zbins)])
        self.cross_networks = nn.ModuleList([single_zbin_transformer(config_dict, is_cross_spectra=True)
                                             for _ in range(self.num_zbins)])

        # Precompute tracer index pairs in spectrum order: (0,0), (0,1), ..., (nt-1,nt-1)
        pairs = [(i1, i2) for i1, i2 in itertools.product(range(self.num_tracers), repeat=2) if i1 <= i2]
        self.register_buffer("_tracer_idx1", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("_tracer_idx2", torch.tensor([p[1] for p in pairs], dtype=torch.long))

        # Indices into the full spectrum list for auto and cross spectra
        auto_indices  = [i for i, p in enumerate(pairs) if p[0] == p[1]]
        cross_indices = [i for i, p in enumerate(pairs) if p[0] != p[1]]
        self.register_buffer("auto_spectrum_indices",  torch.tensor(auto_indices,  dtype=torch.long))
        self.register_buffer("cross_spectrum_indices", torch.tensor(cross_indices, dtype=torch.long))

        # Tracer indices for each auto/cross subset (for parameter slicing)
        self.register_buffer("_auto_tracer_idx",   torch.tensor([p[0] for p in pairs if p[0] == p[1]], dtype=torch.long))
        self.register_buffer("_cross_tracer_idx1", torch.tensor([p[0] for p in pairs if p[0] != p[1]], dtype=torch.long))
        self.register_buffer("_cross_tracer_idx2", torch.tensor([p[1] for p in pairs if p[0] != p[1]], dtype=torch.long))

    def organize_parameters(self, input_params):
        """Organizes input cosmology + bias parameters into separate auto and cross spectrum tensors.

        Args:
            input_params: tensor with shape [batch, total_param_length]. `total_param_length`
                is expected to be `num_cosmo_params + num_nuisance_params*(num_zbins*num_tracers)`

        Returns:
            auto_params:  tensor with shape [batch * num_auto_spectra, num_zbins,
                num_cosmo_params + num_nuisance_params]
            cross_params: tensor with shape [batch * num_cross_spectra, num_zbins,
                num_cosmo_params + 2*num_nuisance_params]
        """
        bsize = input_params.shape[0]

        # Cosmo params: [batch, nc] → [batch, 1, num_zbins, nc]
        cosmo = input_params[:, :self.num_cosmo_params]
        cosmo_expanded = cosmo[:, None, None, :].expand(bsize, -1, self.num_zbins, -1)

        # Nuisance: [batch, nn*nz*nt] → [batch, nz, nt, nn]
        nuisance = input_params[:, self.num_cosmo_params:]
        nuisance = nuisance.reshape(bsize, self.num_nuisance_params, self.num_zbins, self.num_tracers)
        nuisance = nuisance.permute(0, 2, 3, 1)  # [batch, nz, nt, nn]

        # Auto spectra: one nuisance block per spectrum → [batch, num_auto, nz, nc+nn]
        nus_auto = nuisance[:, :, self._auto_tracer_idx, :]  # [batch, nz, num_auto, nn]
        nus_auto = nus_auto.permute(0, 2, 1, 3)              # [batch, num_auto, nz, nn]
        cosmo_auto = cosmo_expanded.expand(bsize, self.num_auto_spectra, self.num_zbins, -1)
        auto_params = torch.cat([cosmo_auto, nus_auto], dim=-1)  # [batch, num_auto, nz, nc+nn]
        auto_params = auto_params.reshape(bsize * self.num_auto_spectra, self.num_zbins,
                                          self.num_cosmo_params + self.num_nuisance_params)

        # Cross spectra: two nuisance blocks per spectrum → [batch, num_cross, nz, nc+2*nn]
        nus1 = nuisance[:, :, self._cross_tracer_idx1, :].permute(0, 2, 1, 3)  # [batch, num_cross, nz, nn]
        nus2 = nuisance[:, :, self._cross_tracer_idx2, :].permute(0, 2, 1, 3)
        cosmo_cross = cosmo_expanded.expand(bsize, self.num_cross_spectra, self.num_zbins, -1)
        cross_params = torch.cat([cosmo_cross, nus1, nus2], dim=-1)  # [batch, num_cross, nz, nc+2*nn]
        cross_params = cross_params.reshape(bsize * self.num_cross_spectra, self.num_zbins,
                                            self.num_cosmo_params + 2 * self.num_nuisance_params)

        return auto_params, cross_params

    def forward(self, input_params, net_idx=None):
        """Passes an input tensor through the network.

        Args:
            input_params: tuple (auto_params, cross_params) as returned by organize_parameters.
            net_idx: if None, run all networks. Otherwise an integer where
                0..num_zbins-1 selects the auto network for z-bin net_idx, and
                num_zbins..2*num_zbins-1 selects the cross network for z-bin (net_idx - num_zbins).
        """
        auto_params, cross_params = input_params

        bsize_auto  = auto_params.shape[0]
        bsize_cross = cross_params.shape[0]
        auto_spectrum_indices  = torch.arange(self.num_auto_spectra,  device=auto_params.device).repeat(bsize_auto  // self.num_auto_spectra)
        cross_spectrum_indices = torch.arange(self.num_cross_spectra, device=cross_params.device).repeat(bsize_cross // self.num_cross_spectra)

        if net_idx is None:
            bsize = bsize_auto // self.num_auto_spectra
            X = torch.zeros((bsize, self.num_spectra, self.num_zbins, self.output_dim), device=auto_params.device)
            for z in range(self.num_zbins):
                auto_out  = self.auto_networks[z](auto_params[:, z],  auto_spectrum_indices)   # [batch*num_auto, output_dim]
                cross_out = self.cross_networks[z](cross_params[:, z], cross_spectrum_indices) # [batch*num_cross, output_dim]
                auto_out  = auto_out.reshape(bsize,  self.num_auto_spectra,  self.output_dim)
                cross_out = cross_out.reshape(bsize, self.num_cross_spectra, self.output_dim)
                X[:, self.auto_spectrum_indices,  z] = auto_out
                X[:, self.cross_spectrum_indices, z] = cross_out
            return X

        is_cross = net_idx >= self.num_zbins
        z = net_idx - self.num_zbins if is_cross else net_idx
        if not is_cross:
            return self.auto_networks[z](auto_params[:, z], auto_spectrum_indices)
        else:
            return self.cross_networks[z](cross_params[:, z], cross_spectrum_indices)
