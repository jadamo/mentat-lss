import torch
import torch.nn as nn
import numpy as np
import yaml, math, os, copy
import itertools
import logging

from mentat_lss.models import blocks
from mentat_lss.models.stacked_transformer import stacked_transformer
from mentat_lss.models.combined_tracer_transformer import combined_tracer_transformer
from mentat_lss.models.analytic_terms import analytic_eft_model
from mentat_lss.models.cov_mlp import stacked_cov_network
from mentat_lss.dataset import pk_galaxy_dataset, cov_matrix_dataset
from mentat_lss.utils import load_config_file, get_parameter_ranges,\
                              normalize_cosmo_params, un_normalize_power_spectrum, \
                              delta_chi_squared, mse_loss, hyperbolic_loss, hyperbolic_chi2_loss, \
                              get_invcov_blocks, get_full_invcov, is_in_hypersphere, symmetric_exp

class ps_emulator():
    """Class defining the neural network emulator."""


    def __init__(self, net_dir:str, mode:str="train", device:torch.device=None):
        """Emulator constructor, initializes the network structure and all supporting data.

        Args:
            net_dir (str): path specifying either the directory or full filepath of the trained emulator to load from.
            if a directory, assumes the config file is called "config.yaml"
            mode (str): whether the emulator should initialize for training, or to load from a previous training run. One 
            of either ["train", "eval"]. Detailt "train"
            device (torch.device): Device to load the emulator on. If None, will attempt to load on any available
            GPU (or mps for macos) device. Default None.

        Raises:
            KeyError: if mode is not correctly specified
            IOError: if no input yaml file was found
        """
        if net_dir.endswith(".yaml"): self.config_dict = load_config_file(net_dir)
        else:                         self.config_dict = load_config_file(os.path.join(net_dir,"config.yaml"))

        self.logger = logging.getLogger('ps_emulator')

        # Infers the number of k-bins, z-bins, tracers, and ells from the training data if not provided
        if np.any([key not in self.config_dict for key in ["num_kbins", "num_ells", "num_zbins", "num_tracers"]]):
            self._load_ps_properties(os.path.join(self.config_dict["input_dir"], self.config_dict["training_dir"]))

        # load dictionary entries into their own class variables
        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])
        # flatten architecture and training sub-dicts into direct attributes
        for key, val in self.config_dict.get("galaxy_ps_emulator", {}).items():
            setattr(self, key, val)
        for key, val in self.config_dict.get("galaxy_ps_training_params", {}).items():
            setattr(self, key, val)

        self._init_device(device, mode)
        self._init_model()
        self._init_loss()

        if mode == "train":
            self.logger.debug("Initializing power spectrum emulator in training mode")
            self._init_fiducial_power_spectrum()
            self._init_inverse_covariance()
            self._diagonalize_covariance()
            self._init_input_normalizations()
            self.galaxy_ps_model.apply(self._init_weights)
            self.galaxy_ps_checkpoint = copy.deepcopy(self.galaxy_ps_model.state_dict())

        elif mode == "eval":
            self.logger.debug("Initializing power spectrum emulator in evaluation mode")
            self.load_trained_model(net_dir)
            self._init_analytic_model()

        else:
            raise KeyError(f"Invalid mode specified! Must be one of ['train', 'eval'] but was {mode}.")

    def _load_ps_properties(self, path):
        """loads the k-bins, z-bins, and effective redshifts of the power spectrum training data from file. 
        This is used to check compatibility of the emulator with the training data, and to load in the effective redshifts for use in the analytic model.

        Args:
            path (str): the directory+filename of the ps_properties.npz file to load
        Raises:
            IOError: if no ps_properties.npz file is found at the given path
        """

        ps_properties = np.load(os.path.join(path, "ps_properties.npz"))
        self.k_emu = ps_properties["k"]
        self.ells  = ps_properties["ells"]
        self.z_eff = ps_properties["z_eff"]
        self.ndens = ps_properties["ndens"]

        self.config_dict["num_kbins"]   = len(self.k_emu)
        self.config_dict["num_ells"]    = len(self.ells)
        self.config_dict["num_zbins"]   = len(self.z_eff)
        self.config_dict["num_tracers"] = len(self.ndens)

        self.logger.info(f"Emulator using {self.config_dict['num_kbins']} k-bins, {self.config_dict['num_ells']} ells, {self.config_dict['num_zbins']} z-bins, and {self.config_dict['num_tracers']} tracers (based on loaded ps_properties.npz file)")

    def load_trained_model(self, path):
        """loads the pre-trained network from file into the current model, as well as all relavent information needed for normalization.
        This function is called by the constructor, but can also be called directly by the user if desired.
        
        Args:
            path: The directory+filename of the trained network to load. 
        """

        self.logger.info(f"loading emulator from {path}")
        self.galaxy_ps_model.eval()
        self.galaxy_ps_model.load_state_dict(torch.load(os.path.join(path,'network_galaxy.params'), 
                                                        weights_only=True, map_location=self.device))

        if not hasattr(self, "k_emu") or not hasattr(self, "ells") or not hasattr(self, "z_eff") or not hasattr(self, "ndens"):
            ps_properties = np.load(os.path.join(path, "ps_properties.npz"))
            self.k_emu = ps_properties["k"]
            self.ells = ps_properties["ells"]
            self.z_eff = ps_properties["z_eff"]
            self.ndens = ps_properties["ndens"]

        input_norm_data = torch.load(os.path.join(path,"input_normalizations.pt"), 
                                     map_location=self.device, weights_only=True)
        self.input_normalizations = input_norm_data[0] # <- in shape expected by networks
        self.required_emu_params  = input_norm_data[1]
        self.emu_param_bounds     = input_norm_data[2]

        output_norm_data = torch.load(os.path.join(path,"output_normalizations.pt"), 
                                      map_location=self.device, weights_only=True)
        self.ps_fid        = output_norm_data[0]
        self.invcov_full   = output_norm_data[1]
        self.invcov_blocks = output_norm_data[2]
        self.sqrt_eigvals  = output_norm_data[3]
        self.Q             = output_norm_data[4]
        self.Q_inv = torch.zeros_like(self.Q, device="cpu")
        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            self.Q_inv[ps, z] = torch.linalg.inv(self.Q[ps, z].to("cpu").to(torch.float64)).to(torch.float32)
        self.Q_inv = self.Q_inv.to(self.device)


    def load_data(self, key:str, data_frac = 1.0, return_dataloader=True, data_dir=""):
        """loads and returns the training / validation / test dataset into memory
        
        Args:
            key: one of ["training", "validation", "testing"] that specifies what type of data-set to laod
            data_frac: fraction of the total data-set to load in. Default 1
            return_dataloader: Determines what object type to return the data as. Default True
                If true: returns data as a pytorch.utils.data.DataLoader object.
                If false: returns data as a pk_galaxy_dataset object.
            data_dir: location of the data-set on disk. Default ""

        Returns:
            data: The desired data-set in either a pk_galaxy_dataset or DataLoader object.

        Raises:
            KeyError: If key is an incorrect value.
            ValueError: If some property of the loaded dataset does not match with the emulator.
        """

        if data_dir != "": dir = data_dir
        else :             dir = self.input_dir+self.training_dir

        if not hasattr(self, "k_emu"):
            self.logger.info("loading ps properties from training set")
            ps_properties = np.load(os.path.join(dir, "ps_properties.npz"))
            self.k_emu = ps_properties["k"]
            self.ells = ps_properties["ells"]
            self.z_eff = ps_properties["z_eff"]
            self.ndens = ps_properties["ndens"]

        if key in ["training", "validation", "testing"]:
            data = pk_galaxy_dataset(dir, key, data_frac)
            data.to(self.device)
            data.normalize_data(self.ps_fid, self.sqrt_eigvals, self.Q)

            data_loader = torch.utils.data.DataLoader(data, batch_size=self.config_dict["batch_size"], shuffle=True)
            self._check_training_set(data)

            if return_dataloader: return data_loader
            else: return data
        else:
            raise KeyError("Invalid value for key! must be one of ['training', 'validation', 'testing']")


    def get_power_spectra(self, params, extrapolate:bool = False, raw_output:bool = False):
        """Gets the full galaxy power spectrum multipoles (emulated and analytically calculated)
        
        Args:
            params: 1D or 2D numpy array, torch Tensor, or dictionary containing a list of cosmology + galaxy bias parameters. 
                if params is a 2D array, this function generates a batch of power spectra simultaniously
            extrapolate (bool): Whether or not to pass through the emulator if the given input parameters are outside the range it was trained on.
                Default False
            raw_output (bool): Whether or not to return the raw network output without undoing normalization. Default False
        
        Returns:
            galaxy_ps (np.array): Emulated galaxy power spectrum multipoles. 
            If raw_output = False, has shape [nps, nz, nk, nl] or [nb, nps, nz, nk, nl]. Else has shape [nb, nps, nz, nk*nl]
        """
        galaxy_ps_emu = self.get_emulated_power_spectrum(params, extrapolate, raw_output)

        if len(galaxy_ps_emu.shape) == 4 and raw_output == False: 
            return galaxy_ps_emu + self.analytic_model.get_analytic_terms(params, self.required_emu_params, self.get_required_analytic_parameters())
        else:
            return galaxy_ps_emu


    def get_emulated_power_spectrum(self, params, extrapolate:bool = False, raw_output:bool = False):
        """Gets the power spectra corresponding to the given input params by passing them though the emulator
        
        Args:
            params: 1D or 2D numpy array, torch Tensor, or dictionary containing a list of cosmology + galaxy bias parameters. 
            if params is a 2D array, this function generates a batch of power spectra simultaniously
            extrapolate (bool): Whether or not to pass through the emulator if the given input parameters are outside the range it was trained on.
            Default False
            raw_output: bool specifying whether or not to return the raw network output without undoing normalization. Default False

        Returns:
            galaxy_ps (np.array): emulated galaxy power spectrum multipoles (P_tree + P_1loop). If given a batch of parameters, has shape [nb, nps, nz, nk, nl]. 
            Otherwise, has shape [nps, nz, nk, nl]. If extrapolate is false and the given input parameters are out of bounds, then this function returns
            an array of all zeros.
        """
        
        self.galaxy_ps_model.eval()
        with torch.no_grad():
            emu_params, skip_emulation = self._check_params(params, extrapolate)
            if skip_emulation and not raw_output and len(params.shape) == 1:
                return np.zeros((self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells))
            elif skip_emulation and not raw_output and len(params.shape) > 1:
                return np.zeros((params.shape[0], self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells))

            galaxy_ps = self.galaxy_ps_model.forward(emu_params) # <- shape [nb, nps, nz, nk*nl]
            
            if raw_output:
                return galaxy_ps

            if self.model_type == "combined_tracer_transformer":
                batch_size = params.shape[0] if len(params.shape) > 1 else 1
                galaxy_ps = galaxy_ps.reshape(batch_size, self.num_spectra, self.num_zbins, self.num_kbins * self.num_ells)
            else:
                galaxy_ps = torch.flatten(galaxy_ps, start_dim=3)
            galaxy_ps = un_normalize_power_spectrum(galaxy_ps, self.ps_fid, self.sqrt_eigvals, self.Q, self.Q_inv)

            if len(params.shape) == 1:
                galaxy_ps = galaxy_ps.view(self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
            else:
                galaxy_ps = galaxy_ps.view(-1, self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)

            return galaxy_ps.to("cpu").detach().numpy()


    def get_required_emu_parameters(self):
        """Returns a list of input parameters needed by the emulator. 
        
        Currently, mentat-lss requires input parameters to be in the same order as given by
        the return value of this function. For example. If the return list is ['h', 'omch2'], you
        should pass in [h, omch2] to get_power_spectra in that order.
        
        Returns:
            required_emu_params (list): list of input cosmology + bias parameters required by the emulator.
        """
        return self.required_emu_params


    def get_required_analytic_parameters(self):
        """Returns a list of input parameters used by our analytic eft model, not directly emulated.
        
        NOTE: These parameters are currently hard-coded.

        Returns:
            required_analytic_params (list): list of input (counterterm + stoch) parameters.
        """
        analytic_params = []
        if 0 in self.ells:  analytic_params.append("counterterm_0")
        if 2 in self.ells:  analytic_params.append("counterterm_2")
        if 4 in self.ells:  analytic_params.append("counterterm_4")
        analytic_params.extend(["counterterm_fog", "P_shot"])
        return analytic_params


    def check_kbins_are_compatible(self, test_kbins:np.array):
        """Tests whether the passed test_kbins is the same as the emulator k-bins

        Args:
            test_kbins (np.array): k-array to check
        Returns:
            is_compatible (bool): Whether or not the given k-bins are compatible
        """
        
        if test_kbins.shape != self.k_emu.shape: return False
        else: return np.allclose(test_kbins, self.k_emu)


    # -----------------------------------------------------------
    # Helper methods: Not meant to be called by the user directly
    # -----------------------------------------------------------

    def _init_device(self, device, mode):
        """Sets emulator device based on machine configuration"""
        self.num_gpus = torch.cuda.device_count()
        if mode == "eval":                      self.device = torch.device("cpu")
        elif device != None:                    self.device = device
        elif self.use_gpu == False:             self.device = torch.device('cpu')
        elif torch.cuda.is_available():         self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else:                                   self.device = torch.device('cpu')


    def _init_model(self):
        """Initializes the networks"""
        self.num_spectra = self.num_tracers + math.comb(self.num_tracers, 2)
        if self.model_type == "stacked_transformer":
            self.galaxy_ps_model = stacked_transformer(self.config_dict).to(self.device)
        elif self.model_type == "combined_tracer_transformer":
            self.galaxy_ps_model = combined_tracer_transformer(self.config_dict).to(self.device)
        else:
            raise KeyError(f"Invalid value for model_type: {self.model_type}")
        

    def _init_analytic_model(self):
        """Initializes object for calculating analytic eft terms"""

        self.analytic_model = analytic_eft_model(self.num_tracers, self.z_eff, self.ells, self.k_emu, self.ndens)


    def _init_input_normalizations(self):
        """Initializes input parameter names and normalization factors
        
        Normalizations are in the shape (low / high bound, net_idx, parameter)
        """

        try:
            cosmo_dict = load_config_file(os.path.join(self.input_dir,self.cosmo_dir))
            param_names, param_bounds = get_parameter_ranges(cosmo_dict)
            self.input_normalizations = torch.Tensor(param_bounds.T).to(self.device)
        except IOError:
            self.input_normalizations = torch.vstack((torch.zeros((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))),
                                                 torch.ones((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))))).to(self.device)
            param_names, param_bounds = [], np.empty((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params), 2))

        # lower_bounds = self.galaxy_ps_model.organize_parameters(input_normalizations[0].unsqueeze(0))
        # upper_bounds = self.galaxy_ps_model.organize_parameters(input_normalizations[1].unsqueeze(0))

        # self.input_normalizations = torch.vstack([lower_bounds, upper_bounds])
        # print(self.input_normalizations.shape, lower_bounds.shape)
        self.required_emu_params = param_names
        self.emu_param_bounds = torch.from_numpy(param_bounds).to(torch.float32).to(self.device)


    def _init_fiducial_power_spectrum(self):
        """Loads the fiducial galaxy and non-wiggle power spectrum for use in normalization"""

        ps_file = self.input_dir+self.training_dir+"ps_fid.npy"
        if os.path.exists(ps_file):
            self.ps_fid = torch.from_numpy(np.load(ps_file)).to(torch.float32).to(self.device)[0]

            # permute input power spectrum if it's a different shape than expected
            if self.ps_fid.shape[3] == self.num_kbins:
                self.ps_fid = torch.permute(self.ps_fid, (0, 1, 3, 2))
            if self.ps_fid.shape[0] == self.num_zbins:
                self.ps_fid = torch.permute(self.ps_fid, (1, 0, 2, 3))
            self.ps_fid = self.ps_fid.reshape(self.num_spectra, self.num_zbins, self.num_kbins * self.num_ells)
        else:
            self.ps_fid = torch.zeros((self.num_spectra, self.num_zbins, self.num_kbins * self.num_ells)).to(self.device)


    def _init_inverse_covariance(self):
        """Loads the inverse data covariance matrix for use in certain loss functions and normalizations"""

        # TODO: Upgrade to handle different number of k-bins for each zbin
        cov_file = self.input_dir+self.training_dir
        # Temporarily store with double percision to increase numerical stability\
        if os.path.exists(cov_file+"cov.dat"):
            cov = torch.load(cov_file+"cov.dat", weights_only=True).to(torch.float64)
        elif os.path.exists(cov_file+"cov.npy"):
            cov = torch.from_numpy(np.load(cov_file+"cov.npy"))
        else:
            self.logger.warning("Could not find covariance matrix! Using identity matrix instead...")
            cov = torch.eye(self.num_spectra*self.num_ells*self.num_kbins).unsqueeze(0)
            cov = cov.repeat(self.num_zbins, 1, 1)  

        self.invcov_blocks = get_invcov_blocks(cov, self.num_spectra, self.num_zbins, self.num_kbins, self.num_ells)
        self.invcov_full   = get_full_invcov(cov, self.num_zbins)


    def _diagonalize_covariance(self):
        """performs an eigenvalue decomposition of the each diagonal block of the inverse covariance matrix
           this function is always performed on cpu in double percision to improve stability"""
        
        self.Q = torch.zeros_like(self.invcov_blocks)
        self.Q_inv = torch.zeros_like(self.invcov_blocks)
        self.sqrt_eigvals = torch.zeros((self.num_spectra, self.num_zbins, self.num_ells*self.num_kbins))

        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            eig, q = torch.linalg.eigh(self.invcov_blocks[ps, z])
            assert torch.all(torch.isnan(q)) == False
            assert torch.all(eig > 0), "ERROR! inverse covariance matrix has negative eigenvalues? Is it positive definite?"

            self.Q[ps, z] = q.real
            self.Q_inv[ps, z] = torch.linalg.inv(q).real
            self.sqrt_eigvals[ps, z] = torch.sqrt(eig)

        # move data to gpu and convert to single percision
        self.invcov_blocks = self.invcov_blocks.to(torch.float32).to(self.device)
        self.invcov_full = self.invcov_full.to(torch.float32).to(self.device)
        self.Q = self.Q.to(torch.float32).to(self.device)
        self.Q_inv = self.Q_inv.to(torch.float32).to(self.device)
        self.sqrt_eigvals = self.sqrt_eigvals.to(torch.float32).to(self.device)


    def _init_loss(self):
        """Defines the loss function to use"""

        if self.loss_type == "chi2":
            self.loss_function = delta_chi_squared
        elif self.loss_type == "mse":
            self.loss_function = mse_loss
        elif self.loss_type == "hyperbolic":
            self.loss_function = hyperbolic_loss
        elif self.loss_type == "hyperbolic_chi2":
            self.loss_function = hyperbolic_chi2_loss
        else:
            raise KeyError("ERROR: Invalid loss function type! Must be one of ['chi2', 'mse', 'hyperbolic', 'hyperbolic_chi2']")


    def _init_weights(self, m):
        """Initializes weights using a specific scheme set in the input yaml file
        
        This function is meant to be called by the constructor only.
        Current options for initialization schemes are ["normal", "He", "xavier"]
        """
        if isinstance(m, nn.Linear):
            if self.weight_initialization == "He":
                nn.init.kaiming_uniform_(m.weight)
            elif self.weight_initialization == "normal":
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
            elif self.weight_initialization == "xavier":
                nn.init.xavier_normal_(m.weight)
            else: # if scheme is invalid, use normal initialization as a substitute
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
        elif isinstance(m, blocks.linear_with_channels):
            m.initialize_params(self.weight_initialization)


    def _init_training_stats(self):
        """initializes training data as nested lists with dims [nps, nz]"""

        if self.model_type == "combined_tracer_transformer":
            num_nets = 2 * self.num_zbins
        else:
            num_nets = self.num_spectra * self.num_zbins

        self.train_loss = [[] for i in range(num_nets)]
        self.valid_loss = [[] for i in range(num_nets)]
        self.train_time = 0. # <- One value for the full emulator, not one for each sub-network


    def _init_optimizer(self):
        """Sets optimization objects, one for each sub-network"""

        if self.model_type == "combined_tracer_transformer":
            num_nets = 2 * self.num_zbins
        else:
            num_nets = self.num_spectra * self.num_zbins

        self.optimizer = [None for i in range(num_nets)]
        self.scheduler = [None for i in range(num_nets)]
        for net_idx in range(num_nets):
            if self.optimizer_type == "Adam":
                if self.model_type == "combined_tracer_transformer":
                    if net_idx < self.num_zbins:
                        net = self.galaxy_ps_model.auto_networks[net_idx]
                    else:
                        net = self.galaxy_ps_model.cross_networks[net_idx - self.num_zbins]
                    self.optimizer[net_idx] = torch.optim.Adam(net.parameters(),
                                                               lr=self.learning_rate)
                else:
                    self.optimizer[net_idx] = torch.optim.Adam(self.galaxy_ps_model.networks[net_idx].parameters(),
                                                               lr=self.learning_rate)
            else:
                raise KeyError("Error! Invalid optimizer type specified!")

            # use an adaptive learning rate
            self.scheduler[net_idx] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer[net_idx],
                                "min", factor=0.1, patience=15)


    def _update_checkpoint(self, net_idx=0, mode="galaxy_ps"):
        """saves current best network to an independent state_dict"""
        if mode == "galaxy_ps":
            new_checkpoint = self.galaxy_ps_model.state_dict()
            if self.model_type == "combined_tracer_transformer":
                if net_idx < self.num_zbins:
                    key_prefix = f"auto_networks.{int(net_idx)}."
                else:
                    key_prefix = f"cross_networks.{int(net_idx - self.num_zbins)}."
            else:
                key_prefix = f"networks.{int(net_idx)}."
            for name in new_checkpoint.keys():
                if key_prefix in name:
                    self.galaxy_ps_checkpoint[name] = new_checkpoint[name]
        else:
            raise NotImplementedError

        self._save_model()


    def _save_model(self):
        """saves the current model state and normalization information to file"""

        save_dir = os.path.join(self.input_dir, self.save_dir)
        training_data_dir = os.path.join(save_dir, "training_statistics")
        # HACK for training on multiple GPUS - need to create parent directory first
        if not os.path.exists(os.path.dirname(os.path.dirname(save_dir))):
            os.mkdir(os.path.dirname(os.path.dirname(save_dir)))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # training statistics
        if not os.path.exists(training_data_dir):
            os.mkdir(training_data_dir)

        if self.model_type == "combined_tracer_transformer":
            num_nets = 2*self.num_zbins
        else:
            num_nets = self.num_zbins * self.num_spectra
        for net_idx in range(num_nets):
            torch.save({"train loss" : torch.Tensor(self.train_loss[net_idx]), 
                        "valid loss" : torch.Tensor(self.valid_loss[net_idx]),
                        "train time" : torch.Tensor([self.train_time])},
                        os.path.join(training_data_dir, "train_data_"+str(net_idx)+".dat"))
        
        # configuration data
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, sort_keys=False, default_flow_style=False)
        if hasattr(self, "k_emu"):
            np.savez(os.path.join(save_dir, "ps_properties.npz"), k=self.k_emu, ells=self.ells, z_eff=self.z_eff, ndens=self.ndens)
        else:
            self.logger.warning("power spectrum properties not initialized!")

        # data related to input normalization
        input_files = [self.input_normalizations, self.required_emu_params, self.emu_param_bounds]
        torch.save(input_files, os.path.join(save_dir, "input_normalizations.pt"))
        with open(os.path.join(save_dir, "param_names.txt"), "w") as outfile:
            yaml.dump(self.get_required_emu_parameters(), outfile, sort_keys=False, default_flow_style=False)

        # data related to output normalization
        output_files = [self.ps_fid, self.invcov_full, self.invcov_blocks, self.sqrt_eigvals, self.Q]
        torch.save(output_files, os.path.join(save_dir, "output_normalizations.pt"))

        # Finally, the actual model parameters
        torch.save(self.galaxy_ps_checkpoint, os.path.join(save_dir, 'network_galaxy.params'))


    def _check_params(self, params, extrapolate=False):
        """checks that input parameters are in the expected format and within the specified boundaries"""
        skip_emulation = False

        if isinstance(params, torch.Tensor): 
            params = params.to(self.device)
        elif isinstance(params, np.ndarray): 
            params = torch.from_numpy(params).to(torch.float32).to(self.device)
        else:
            raise TypeError(f"invalid type for variable params ({type(params)})")
        
        if params.dim() == 1: params = params.unsqueeze(0)

        if params.shape[1] > len(self.required_emu_params):
            params = params[:, :len(self.required_emu_params)]

        # TODO: Better handling with batch of parameters
        # Right now, this if-statement will trigger if any of the batch of parameters
        # are out of bounds
        if (self.sampling_type == "hypercube" and \
            torch.any(params < self.input_normalizations[0]) or \
            torch.any(params > self.input_normalizations[1])) or \
           (self.sampling_type == "hypersphere" and \
            not torch.any(is_in_hypersphere(self.emu_param_bounds, params)[0])):
            if extrapolate:
                self.logger.warning("Input parameters out of bounds! Emulator output will be untrustworthy")
            else: 
                self.logger.info("Input parameters out of bounds! Skipping emulation...")
                skip_emulation = True

        norm_params = normalize_cosmo_params(params, self.input_normalizations)
        org_norm_params = self.galaxy_ps_model.organize_parameters(norm_params)
        return org_norm_params, skip_emulation

    def _check_training_set(self, data:pk_galaxy_dataset):
        """checks that loaded-in data for training / validation / testing is compatable with the given network config
        
        Raises:
            ValueError: If a given property of the training set does not match with the emulator.
        """

        if len(data.cosmo_params) != self.num_cosmo_params:
            raise ValueError("num_cosmo_params mismatch with training dataset! {:d} vs {:d}".format(len(data.cosmo_params), self.num_cosmo_params))
        if len(data.bias_params) != self.num_nuisance_params*self.num_tracers*self.num_zbins:
            raise ValueError("num_nuisance_params mismatch with training dataset! {:d} vs {:d}".format(len(data.bias_params), self.num_nuisance_params*self.num_tracers*self.num_zbins))
        if data.num_spectra != self.num_spectra:
            raise(ValueError("num_spectra (derived from num_tracers) mismatch with training dataset! {:d} vs {:d}".format(data.num_spectra, self.num_spectra)))
        if data.num_zbins != self.num_zbins:
            raise(ValueError("num_ells mismatch with training dataset! {:d} vs {:d}".format(data.num_zbins, self.num_zbins)))
        if data.num_ells != self.num_ells:
            raise(ValueError("num_ells mismatch with training dataset! {:d} vs {:d}".format(data.num_ells, self.num_ells)))
        if data.num_kbins != self.num_kbins:
            raise(ValueError("num_ells mismatch with training dataset! {:d} vs {:d}".format(data.num_kbins, self.num_kbins)))
        

# --------------------------------------------------------------------------
# extra helper function (TODO: Find a better place for this)
# --------------------------------------------------------------------------
def compile_multiple_device_training_results(save_dir:str, config_dir:str, num_gpus:int):
    """takes networks saved on seperate ranks and combines them to the same format as when training on one device
    
    Args:
        save_dir (string): base save directory, where each rank was saved in its own sub-directory
        config_dir (string): path+name of the original network config file
        num_gpus (int): number of gpus to compile results of
    Returns:
        full_emulator (ps_emulator): emulator object with all training data combined together.
    """

    full_emulator = ps_emulator(config_dir, "train")
    full_emulator.galaxy_ps_model.eval()

    if full_emulator.model_type == "combined_tracer_transformer":
        net_idx = torch.Tensor(list(range(2*full_emulator.num_zbins))).to(int)
        num_nets = 2*full_emulator.num_zbins
    else:
        net_idx = torch.Tensor(list(itertools.product(range(full_emulator.num_spectra), range(full_emulator.num_zbins)))).to(int)
        num_nets = full_emulator.num_zbins * full_emulator.num_spectra
    split_indices = net_idx.chunk(num_gpus)

    full_emulator._init_training_stats()
    # full_emulator.train_loss = torch.zeros((num_nets, full_emulator.num_epochs))
    # full_emulator.valid_loss = torch.zeros((num_nets, full_emulator.num_epochs))
    # full_emulator.train_time = 0.
    for n in range(num_gpus):
        sub_dir = "rank_"+str(n)
        seperate_network = ps_emulator(os.path.join(save_dir,sub_dir), "eval")

        # power spectrum properties used by analytic_terms.py
        if n == 0:
            ps_properties = np.load(os.path.join(save_dir, sub_dir, "ps_properties.npz"))
            full_emulator.k_emu = ps_properties["k"]
            full_emulator.ells = ps_properties["ells"]
            full_emulator.z_eff = ps_properties["z_eff"]
            full_emulator.ndens = ps_properties["ndens"]

        # galaxy power spectrum networks
        for idx in split_indices[n]:
            if full_emulator.model_type == "combined_tracer_transformer":
                net_idx  = idx
                is_cross = net_idx >= full_emulator.num_zbins
                z        = net_idx - full_emulator.num_zbins if is_cross else net_idx
                if is_cross:
                    full_emulator.galaxy_ps_model.cross_networks[z] = seperate_network.galaxy_ps_model.cross_networks[z]
                else:
                    full_emulator.galaxy_ps_model.auto_networks[z] = seperate_network.galaxy_ps_model.auto_networks[z]
            else:
                net_idx = (idx[1] * full_emulator.num_spectra) + idx[0]
                full_emulator.galaxy_ps_model.networks[net_idx] = seperate_network.galaxy_ps_model.networks[net_idx]

            train_data = torch.load(os.path.join(save_dir,sub_dir,"training_statistics/train_data_"+str(int(net_idx))+".dat"), weights_only=True)
            full_emulator.train_loss[net_idx] = train_data["train loss"].tolist()
            full_emulator.valid_loss[net_idx] = train_data["valid loss"].tolist()
            full_emulator.train_time = train_data["train time"]

    full_emulator.galaxy_ps_checkpoint = copy.deepcopy(full_emulator.galaxy_ps_model.state_dict())

    return full_emulator


# --------------------------------------------------------------------------
# Covariance matrix emulator
# --------------------------------------------------------------------------
class cov_emulator():
    """Class defining the covariance matrix emulator.

    Wraps a stacked_cov_network (one sub-network per redshift bin) for both
    training and evaluation. Parameter bounds are read from the same cosmology
    config YAML used by ps_emulator, and normalization is applied externally
    (consistently with the ps_emulator pattern).

    In evaluation mode the primary user-facing method is get_covariance_matrix.
    In training mode, use load_data to retrieve DataLoaders and then call the
    functions in training_loops.py.
    """

    def __init__(self, net_dir:str, mode:str="train", device:torch.device=None):
        """Emulator constructor, initializes the network and all supporting data.

        Args:
            net_dir (str): path to either the directory or full filepath of the emulator
                config. If a directory, assumes the config file is named "config.yaml".
            mode (str, optional): one of ["train", "eval"]. Defaults to "train".
            device (torch.device, optional): device to run on. If None, selects any
                available GPU (or MPS on macOS), falling back to CPU. Defaults to None.

        Raises:
            KeyError: if mode is not one of ["train", "eval"]
            IOError: if no config yaml file is found at net_dir
        """
        if net_dir.endswith(".yaml"):
            self.config_dict = load_config_file(net_dir)
        else:
            self.config_dict = load_config_file(os.path.join(net_dir, "config.yaml"))

        self.logger = logging.getLogger("cov_emulator")

        # infer num_zbins and num_tracers from cov_properties.npz if not in config
        if np.any([key not in self.config_dict for key in ["num_zbins", "num_tracers"]]):
            self._load_cov_properties(os.path.join(self.config_dict["input_dir"], self.config_dict["training_dir"]))

        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])
        # flatten architecture and training sub-dicts into direct attributes
        for key, val in self.config_dict.get("covariance_emulator", {}).items():
            setattr(self, key, val)
        for key, val in self.config_dict.get("covariance_training_params", {}).items():
            setattr(self, key, val)

        self._init_device(device, mode)
        self._init_model()

        if mode == "train":
            self.logger.debug("Initializing covariance emulator in training mode")
            self._init_input_normalizations()
            # norm values are computed lazily from the training data on first load_data call
            self.norm_pos = 0.
            self.norm_neg = 0.
            self.cov_model.apply(self._init_weights)
            self.cov_checkpoint = copy.deepcopy(self.cov_model.state_dict())

        elif mode == "eval":
            self.logger.debug("Initializing covariance emulator in evaluation mode")
            self.load_trained_model(net_dir)

        else:
            raise KeyError(f"Invalid mode specified! Must be one of ['train', 'eval'] but was {mode}.")


    def _load_cov_properties(self, path:str):
        """Loads survey geometry (z-bins and tracers) from cov_properties.npz.

        Sets num_zbins and num_tracers in config_dict so they are available to the
        model constructor and the setattr loop. Analogous to ps_emulator._load_ps_properties.

        Args:
            path (str): directory containing cov_properties.npz

        Raises:
            IOError: if no cov_properties.npz file is found at path
        """
        props_path = os.path.join(path, "cov_properties.npz")
        if not os.path.exists(props_path):
            raise IOError(f"No cov_properties.npz file found at {props_path}")

        cov_props = np.load(props_path)
        self.z_eff = cov_props["z_eff"]
        self.ndens = cov_props["ndens"]

        self.config_dict["num_zbins"]   = len(self.z_eff)
        self.config_dict["num_tracers"] = len(self.ndens)

        self.logger.info(
            f"Covariance emulator using {self.config_dict['num_zbins']} z-bins and "
            f"{self.config_dict['num_tracers']} tracers (based on cov_properties.npz)")


    def load_trained_model(self, path:str):
        """Loads a pre-trained network from file.

        This is called automatically by the constructor in eval mode, but can also
        be called directly to reload a checkpoint.

        Args:
            path (str): directory containing network_cov.params, normalizations.pt,
                and input_normalizations.pt
        """
        self.logger.info(f"loading covariance emulator from {path}")

        if not hasattr(self, "z_eff") or not hasattr(self, "ndens"):
            cov_props = np.load(os.path.join(path, "cov_properties.npz"))
            self.z_eff = cov_props["z_eff"]
            self.ndens = cov_props["ndens"]

        self.cov_model.eval()
        self.cov_model.load_state_dict(
            torch.load(os.path.join(path, "network_cov.params"),
                       weights_only=True, map_location=self.device))

        norm_data = torch.load(os.path.join(path, "normalizations.pt"),
                               map_location="cpu", weights_only=True)
        self.norm_pos = norm_data[0]
        self.norm_neg = norm_data[1]

        input_norm_data = torch.load(os.path.join(path, "input_normalizations.pt"),
                                     map_location=self.device, weights_only=True)
        self.input_normalizations = input_norm_data[0]
        self.required_emu_params  = input_norm_data[1]


    def load_data(self, type:str, frac:float=1., return_dataloader:bool=True):
        """Loads a covariance matrix dataset from disk.

        For the training set, the matrix normalization values (norm_pos, norm_neg)
        are computed from the data on the first call and stored on the emulator for
        use with validation/test sets.

        Args:
            type (str): one of ["training", "validation", "testing"]
            frac (float, optional): fraction of the dataset to load. Defaults to 1.0.
            return_dataloader (bool, optional): if True, wraps the dataset in a
                DataLoader. Defaults to True.

        Returns:
            loader (DataLoader or cov_matrix_dataset): the loaded data
        """
        data_dir = os.path.join(self.input_dir, self.training_dir)

        dataset = cov_matrix_dataset(data_dir, type, frac,
                                     gaussian_only=self.train_gaussian_only,
                                     pos_norm=self.norm_pos, neg_norm=self.norm_neg)

        # store matrix normalization values from the training set for reuse
        if type.lower() == "training":
            self.norm_pos = dataset.norm_pos
            self.norm_neg = dataset.norm_neg

        dataset.to(self.device)

        if return_dataloader:
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataset


    def get_covariance_matrix(self, params, z_idx:int=0, raw:bool=False):
        """Uses the emulator to return a covariance matrix for a given redshift bin.

        Args:
            params (np.ndarray or torch.Tensor): 1D array of input cosmology + bias
                parameters in the same flat ordering used during training.
            z_idx (int, optional): index of the redshift bin to evaluate. Defaults to 0.
            raw (bool, optional): if True, returns the raw Cholesky factor without
                reversing pre-processing. Defaults to False.

        Returns:
            matrix (np.ndarray or torch.Tensor): if raw is False, returns the full
                symmetric covariance matrix as a numpy array with shape
                (output_dim, output_dim). If raw is True, returns the Tensor
                Cholesky factor.

        Raises:
            TypeError: if params is not a numpy array or torch Tensor
        """
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params).to(torch.float32)
        elif not isinstance(params, torch.Tensor):
            raise TypeError(f"params must be a numpy array or torch.Tensor, but got {type(params)}")

        params = params.to(self.device)
        if params.dim() == 1:
            params = params.unsqueeze(0)

        org_params  = self.cov_model.organize_parameters(params)
        norm_params = normalize_cosmo_params(org_params, self.input_normalizations)

        with torch.no_grad():
            matrix = self.cov_model(norm_params, z_idx=z_idx)

        if raw:
            return matrix

        matrix = symmetric_exp(matrix, self.norm_pos, self.norm_neg)
        matrix = matrix.squeeze(0).to("cpu").detach().numpy().astype(np.float64)
        matrix = np.matmul(matrix, matrix.T)
        return matrix


    def _init_device(self, device:torch.device, mode:str):
        """Selects and stores the compute device.

        Args:
            device (torch.device): user-specified device, or None for auto-selection
            mode (str): "train" or "eval"
        """
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.logger.debug(f"using device: {self.device}")


    def _init_model(self):
        """Builds the stacked_cov_network and moves it to the selected device."""
        self.cov_model = stacked_cov_network(self.config_dict).to(self.device)


    def _init_input_normalizations(self):
        """Reads parameter bounds from the cosmology config and organizes them per zbin.

        Follows the same pattern as ps_emulator._init_input_normalizations.
        The resulting input_normalizations tensor has shape
        (2, num_zbins, num_cosmo_params + num_nuisance_params * num_tracers),
        where index 0 is the lower bound and index 1 is the upper bound.
        """
        cosmo_dict = load_config_file(os.path.join(self.input_dir, self.cosmo_dir))
        param_names, param_bounds = get_parameter_ranges(cosmo_dict)

        flat_bounds = torch.Tensor(param_bounds.T).to(self.device)  # shape (2, n_params)

        # organize flat bounds into per-zbin shape using the same parameter
        # ordering convention as stacked_cov_network.organize_parameters
        lower = self.cov_model.organize_parameters(flat_bounds[0].unsqueeze(0)).squeeze(0)
        upper = self.cov_model.organize_parameters(flat_bounds[1].unsqueeze(0)).squeeze(0)
        self.input_normalizations = torch.stack([lower, upper])  # (2, num_zbins, per_net_dim)

        self.required_emu_params = param_names


    def _init_weights(self, m):
        """Initializes layer weights according to the scheme in config_dict.

        Args:
            m (nn.Module): layer to initialize
        """
        if isinstance(m, nn.Linear):
            scheme = self.weight_initialization
            if scheme == "He":
                nn.init.kaiming_uniform_(m.weight)
            elif scheme == "normal":
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)


    def _init_training_stats(self):
        """Initializes per-zbin training loss history lists."""
        self.train_loss = [[] for _ in range(self.num_zbins)]
        self.valid_loss = [[] for _ in range(self.num_zbins)]
        self.train_time = 0.


    def _init_optimizer(self):
        """Sets up one Adam optimizer and ReduceLROnPlateau scheduler per zbin."""
        self.optimizer = []
        self.scheduler = []
        for z in range(self.num_zbins):
            if self.optimizer_type == "Adam":
                opt = torch.optim.Adam(self.cov_model.networks[z].parameters(),
                                       lr=self.learning_rate)
            else:
                raise KeyError(f"Invalid optimizer type! Must be 'Adam', but got {self.optimizer_type}")
            self.optimizer.append(opt)
            self.scheduler.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.1, patience=15))


    def _update_checkpoint(self, z_idx:int):
        """Saves the current state of the network for zbin z_idx as the best checkpoint.

        Args:
            z_idx (int): index of the redshift bin whose checkpoint to update
        """
        new_state = self.cov_model.state_dict()
        for name in new_state.keys():
            if f"networks.{z_idx}." in name:
                self.cov_checkpoint[name] = new_state[name]
        self._save_model()


    def _save_model(self):
        """Saves the model, normalizations, and training statistics to disk."""
        save_dir = os.path.join(self.input_dir, self.save_dir)
        training_data_dir = os.path.join(save_dir, "training_statistics")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(training_data_dir):
            os.makedirs(training_data_dir)

        # per-zbin training statistics
        for z in range(self.num_zbins):
            if len(self.train_loss[z]) > 0:
                training_data = torch.vstack([torch.Tensor(self.train_loss[z]),
                                              torch.Tensor(self.valid_loss[z])])
                torch.save(training_data,
                           os.path.join(training_data_dir, f"train_data_z{z}.dat"))

        # configuration
        with open(os.path.join(save_dir, "config.yaml"), "w") as outfile:
            yaml.dump(dict(self.config_dict), outfile, sort_keys=False, default_flow_style=False)

        # survey geometry
        if hasattr(self, "z_eff") and hasattr(self, "ndens"):
            np.savez(os.path.join(save_dir, "cov_properties.npz"),
                     z_eff=self.z_eff, ndens=self.ndens)
        else:
            self.logger.warning("cov_properties not initialized — cov_properties.npz not saved.")

        # matrix normalization values
        torch.save([self.norm_pos, self.norm_neg],
                   os.path.join(save_dir, "normalizations.pt"))

        # input parameter normalization values
        torch.save([self.input_normalizations, self.required_emu_params],
                   os.path.join(save_dir, "input_normalizations.pt"))

        # network weights (best checkpoint)
        torch.save(self.cov_checkpoint, os.path.join(save_dir, "network_cov.params"))
