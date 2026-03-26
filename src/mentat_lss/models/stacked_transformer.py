import torch
import torch.nn as nn
import math
import itertools

import mentat_lss.models.blocks as blocks

class single_transformer(nn.Module):
    """Class defining a single independent transformer network"""

    def __init__(self, config_dict, is_cross_spectra:bool):
        """Initializes an individual network, responsible for outputting a portion of the full model vector. The user is not meant to call this function directly.

        Args:
            config_dict: input dictionary with various network architecture options.
            is_cross_spectra: specifies whether the given network is responsible for outputting an auto power spectrum or cross spectrum.
                Either case has slightly different input parameter sizes.
        """
        super().__init__()

        # TODO: Allow specification of activation function        
        self.num_ells = config_dict["num_ells"]
        self.num_kbins = config_dict["num_kbins"]
        self.num_nuisance_params = config_dict["num_nuisance_params"]

        # size of input depends on wether or not the network is for the crosss spectra
        self.is_cross_spectra = is_cross_spectra
        if not is_cross_spectra:
            self.input_dim = config_dict["num_cosmo_params"] + config_dict["num_nuisance_params"]
        else:
            self.input_dim = config_dict["num_cosmo_params"] + (2 * config_dict["num_nuisance_params"])
        
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
                                        "batch",
                                        config_dict["galaxy_ps_emulator"]["use_skip_connection"]))
        
        # expand mlp section output
        split_dim = config_dict["galaxy_ps_emulator"]["split_dim"]
        split_size = config_dict["galaxy_ps_emulator"]["split_size"]
        embedding_dim = split_size*split_dim
        self.embedding_layer = nn.Linear(self.output_dim, embedding_dim)

        # do one transformer block per z-bin for now
        num_heads = config_dict["galaxy_ps_emulator"].get("num_heads", 1)
        self.transformer_blocks = nn.Sequential()
        for i in range(config_dict["galaxy_ps_emulator"]["num_transformer_blocks"]):
            self.transformer_blocks.add_module("Transformer"+str(i+1),
                    blocks.block_transformer_encoder(embedding_dim, split_dim, 0.1, num_heads))
            self.transformer_blocks.add_module("Activation"+str(i+1),
                    blocks.activation_function(embedding_dim))

        self.output_layer = nn.Linear(embedding_dim, self.output_dim)

    def forward(self, input_params:torch.Tensor):
        """Passes an input tensor through the network"""

        if not self.is_cross_spectra:
            input_params = input_params[:, :-self.num_nuisance_params]
        
        X = self.input_layer(input_params)
        X = self.mlp_blocks(X)
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        X = self.output_layer(X)

        return X

class stacked_transformer(nn.Module):
    """Class defining a stack of single_transformer objects, one for each portion of the power spectrum output"""

    def __init__(self, config_dict):
        """Initializes a group of single_transformer based on the input dictionary.
        
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
            for isample1, isample2 in itertools.product(range(self.num_tracers), repeat=2):
                if isample1 > isample2: continue
                self.networks.append(single_transformer(config_dict, (isample1 != isample2)))

    def organize_parameters(self, input_params):
        """Organizes input cosmology + bias parameters into a form the rest of the network expects
        
        Args:
            input_params: tensor of input parameters with shape [batch, num_cosmo_params*(num_nuisance_params*num_zbins*num_tracers)]
        Returns:
            organized_params: tensor of input parameters with shape [batch, num_spectra*num_zbins, num_cosmo_params + (2*self.num_nuisance_params)].
                The bias parameters are split corresponding to their respective redshift / tracer bin
        """

        # parameters shape is (b, nz*nps, num_cosmo*2*num_nuisance)
        organized_params = torch.zeros((input_params.shape[0],
                                       self.num_spectra * self.num_zbins, 
                                       self.num_cosmo_params + (2*self.num_nuisance_params)),
                                       device=input_params.device)

        # fill cosmology parameters (the same for every bin)
        organized_params[:,:, :self.num_cosmo_params] = input_params[:, :self.num_cosmo_params].unsqueeze(1)
        # fill bias params
        # ordering is [params for tracer 1, params for tracer 2]
        iter = 0
        for z in range(self.num_zbins):
            for isample1, isample2 in itertools.product(range(self.num_tracers), repeat=2):
                if isample1 > isample2: continue
                
                idx_1 = (z*self.num_tracers) + isample1
                idx_2 = (z*self.num_tracers) + isample2
                iterate = self.num_tracers*self.num_zbins

                organized_params[:, iter, self.num_cosmo_params:self.num_cosmo_params+self.num_nuisance_params] \
                    = input_params[:, self.num_cosmo_params+idx_1::iterate]
                organized_params[:, iter, self.num_cosmo_params+self.num_nuisance_params:self.num_cosmo_params+2*self.num_nuisance_params] \
                    = input_params[:, self.num_cosmo_params+idx_2::iterate]
                iter+=1

        return organized_params

    def forward(self, input_params, net_idx = None):
        """Passes an input tensor through the network"""
        
        # feed parameters through all sub-networks
        if net_idx == None:
            X = torch.zeros((input_params.shape[0], self.num_spectra, self.num_zbins, self.output_dim), device=input_params.device)
            
            for (z, ps) in itertools.product(range(self.num_zbins), range(self.num_spectra)):
                idx = (z * self.num_spectra) + ps
                X[:, ps, z] = self.networks[idx](input_params[:,idx])
    
        # feed parameters through an individual sub-network (used in training)
        else:
            X = self.networks[net_idx](input_params[:,net_idx])

        return X
        