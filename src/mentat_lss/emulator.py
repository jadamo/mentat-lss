import torch
import torch.nn as nn
import numpy as np
import yaml, math, os, copy
import itertools
import logging

from mentat_lss.models import blocks
from mentat_lss.models.stacked_mlp import stacked_mlp
from mentat_lss.models.stacked_transformer import stacked_transformer
from mentat_lss.models.analytic_terms import analytic_eft_model
from mentat_lss.dataset import pk_galaxy_dataset
from mentat_lss.utils import load_config_file, get_parameter_ranges,\
                              normalize_cosmo_params, un_normalize_power_spectrum, \
                              delta_chi_squared, mse_loss, hyperbolic_loss, hyperbolic_chi2_loss, \
                              get_invcov_blocks, get_full_invcov, is_in_hypersphere

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

        # load dictionary entries into their own class variables
        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

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

        input_norm_data = torch.load(os.path.join(path,"input_normalizations.pt"), 
                                     map_location=self.device, weights_only=True)
        self.input_normalizations = input_norm_data[0] # <- in shape expected by networks
        self.required_emu_params  = input_norm_data[1]
        self.emu_param_bounds     = input_norm_data[2]

        ps_properties = np.load(os.path.join(path, "ps_properties.npz"))
        self.k_emu = ps_properties["k"]
        self.ells = ps_properties["ells"]
        self.z_eff = ps_properties["z_eff"]
        self.ndens = ps_properties["ndens"]

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
            self.logger.info("loading kbins from training set")
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

            galaxy_ps = un_normalize_power_spectrum(torch.flatten(galaxy_ps, start_dim=3), self.ps_fid, self.sqrt_eigvals, self.Q, self.Q_inv)

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
        if self.model_type == "stacked_mlp":
            self.galaxy_ps_model = stacked_mlp(self.config_dict).to(self.device)
        elif self.model_type == "stacked_transformer":
            self.galaxy_ps_model = stacked_transformer(self.config_dict).to(self.device)
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
            input_normalizations = torch.Tensor(param_bounds.T).to(self.device)
        except IOError:
            input_normalizations = torch.vstack((torch.zeros((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))),
                                                 torch.ones((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params))))).to(self.device)
            param_names, param_bounds = [], np.empty((self.num_cosmo_params + (self.num_tracers*self.num_zbins*self.num_nuisance_params), 2))

        lower_bounds = self.galaxy_ps_model.organize_parameters(input_normalizations[0].unsqueeze(0))
        upper_bounds = self.galaxy_ps_model.organize_parameters(input_normalizations[1].unsqueeze(0))
        self.input_normalizations = torch.vstack([lower_bounds, upper_bounds])
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

        self.train_loss = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.valid_loss = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.train_time = 0.


    def _init_optimizer(self):
        """Sets optimization objects, one for each sub-network"""

        self.optimizer = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        self.scheduler = [[[] for i in range(self.num_zbins)] for j in range(self.num_spectra)]
        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            net_idx = (z * self.num_spectra) + ps
            if self.optimizer_type == "Adam":
                self.optimizer[ps][z] = torch.optim.Adam(self.galaxy_ps_model.networks[net_idx].parameters(), 
                                                         lr=self.galaxy_ps_learning_rate)
            else:
                raise KeyError("Error! Invalid optimizer type specified!")

            # use an adaptive learning rate
            self.scheduler[ps][z] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer[ps][z],
                                    "min", factor=0.1, patience=15)


    def _update_checkpoint(self, net_idx=0, mode="galaxy_ps"):
        """saves current best network to an independent state_dict"""
        if mode == "galaxy_ps":
            new_checkpoint = self.galaxy_ps_model.state_dict()
            for name in new_checkpoint.keys():
                if "networks."+str(net_idx) in name:
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

        for (ps, z) in itertools.product(range(self.num_spectra), range(self.num_zbins)):
            training_data = torch.vstack([torch.Tensor(self.train_loss[ps][z]), 
                                          torch.Tensor(self.valid_loss[ps][z]),
                                          torch.Tensor([self.train_time]*len(self.valid_loss[ps][z]))])
            torch.save(training_data, os.path.join(training_data_dir, "train_data_"+str(ps)+"_"+str(z)+".dat"))
        
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

        org_params = self.galaxy_ps_model.organize_parameters(params)

        # TODO: Better handling with batch of parameters
        # Right now, this if-statement will trigger if any of the batch of parameters
        # are out of bounds
        if (self.sampling_type == "hypercube" and \
            torch.any(org_params < self.input_normalizations[0]) or \
            torch.any(org_params > self.input_normalizations[1])) or \
           (self.sampling_type == "hypersphere" and \
            not torch.any(is_in_hypersphere(self.emu_param_bounds, params)[0])):
            if extrapolate:
                self.logger.warning("Input parameters out of bounds! Emulator output will be untrustworthy")
            else: 
                self.logger.info("Input parameters out of bounds! Skipping emulation...")
                skip_emulation = True

        norm_params = normalize_cosmo_params(org_params, self.input_normalizations)
        return norm_params, skip_emulation

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

    net_idx = torch.Tensor(list(itertools.product(range(full_emulator.num_spectra), range(full_emulator.num_zbins)))).to(int)
    split_indices = net_idx.chunk(num_gpus)

    full_emulator.train_loss = torch.zeros((full_emulator.num_spectra, full_emulator.num_zbins, full_emulator.num_epochs))
    full_emulator.valid_loss = torch.zeros((full_emulator.num_spectra, full_emulator.num_zbins, full_emulator.num_epochs))
    full_emulator.train_time = 0.
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
        for (ps, z) in split_indices[n]:
            net_idx = (z * full_emulator.num_spectra) + ps
            full_emulator.galaxy_ps_model.networks[net_idx] = seperate_network.galaxy_ps_model.networks[net_idx]

            train_data = torch.load(os.path.join(save_dir,sub_dir,"training_statistics/train_data_"+str(int(ps))+"_"+str(int(z))+".dat"), weights_only=True)
            epochs = train_data.shape[1]

            full_emulator.train_loss[ps, z, :epochs] = train_data[0,:]
            full_emulator.valid_loss[ps, z, :epochs] = train_data[1,:]
            full_emulator.train_time = train_data[2,0]

    full_emulator.galaxy_ps_checkpoint = copy.deepcopy(full_emulator.galaxy_ps_model.state_dict())
    
    return full_emulator
