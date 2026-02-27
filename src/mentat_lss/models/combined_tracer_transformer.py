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

    def forward(self, input_params:torch.Tensor):
        """Passes an input tensor through the network"""
        
        X = self.input_layer(input_params)
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
        organized_params = torch.zeros((bsize * self.num_spectra,
                                        self.num_zbins,
                                        self.num_cosmo_params + (2 * self.num_nuisance_params)),
                                        device=input_params.device)

        # for each original batch sample and each spectrum, build an output batch entry
        iterate_size = self.num_tracers*self.num_zbins
        for bi in range(bsize):
            iter = 0
            for isample1, isample2 in itertools.product(range(self.num_tracers), repeat=2):
                if isample1 > isample2: continue
                out_idx = bi * self.num_spectra + iter

                # fill cosmology (same for all z)
                organized_params[out_idx, :, :self.num_cosmo_params] = input_params[bi, :self.num_cosmo_params]

                for z in range(self.num_zbins):
                    idx_1 = z * self.num_tracers + isample1
                    idx_2 = z * self.num_tracers + isample2

                    # each slice with step `iterate` selects the sequence of nuisance params
                    # for this (z,tracer) across all nuisance-parameter slots
                    organized_params[out_idx, z, self.num_cosmo_params:self.num_cosmo_params + self.num_nuisance_params] = \
                                     input_params[bi, self.num_cosmo_params + idx_1::iterate_size]

                    organized_params[out_idx, z, self.num_cosmo_params + self.num_nuisance_params:
                                     self.num_cosmo_params + 2 * self.num_nuisance_params] = \
                                     input_params[bi, self.num_cosmo_params + idx_2::iterate_size]
                iter += 1

        return organized_params

    def forward(self, input_params, net_idx = None):
        """Passes an input tensor through the network"""
        
        # feed parameters through all sub-networks
        if net_idx == None:
            X = torch.zeros((input_params.shape[0], self.num_zbins, self.output_dim), device=input_params.device)
            
            for z in range(self.num_zbins):
                X[:, z] = self.networks[z](input_params[:,z])
    
        # feed parameters through an individual sub-network (used in training)
        else:
            X = self.networks[net_idx](input_params[:,net_idx])

        return X
        