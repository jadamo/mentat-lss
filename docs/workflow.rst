.. _workflow:

Workflow Example
================

This page walks through an expected workflow using `mentat-lss`, split up into **generating a training set**, **training the network**, and **testing / using the network**

Generating a Training Set
-------------------------

This stage is where you decide what power spectrum model you want to emulate, and the corresponding
parameter ranges it will be valid in. Assuming you have access to ``ps_1loop``, we have provided a script in 
``scripts/make_training_set_ps_1loop.py`` that will attempt to create a training set in the right format for you.

If you are creating your own training set, the following are the required ingredients and format you should use:

* Cosmology and survey parameters config files. We provide some example files in the ``configs`` directory.
* Your data should be stored in `pk-training.npz`, `pk-validation.npz`, and `pk-testing.npz` files, each of which have:
    * a params array with shape ``[N, num_params]``.
    * a galaxy power spectrum array with shape ``[N, num_auto_plus_cross_spectra, num_redshift_bins, num_kbins, num_ells]``.
* You should have a `kbins.npz` file with the corresponding k-bin centers in a numpy array.
* You should have a `cov.npy` file with a valid covariance matrix (ex: from `CovaPT` or `TheCov`).

Training the Emulator
---------------------

One you have a training set, you'll need to provide a configuration file specifying
the specific network architecture you want to use, as well as all relavent hyperparameters. 
The following is an example of such a file.

.. code-block:: yaml

    # directory that all other paths are relative to
    input_dir: /home/u12/jadamo/SPHEREx/mentat-lss/
    # Where to save the network
    save_dir: ./emulators/stacked_transformer_2t_2z_hypersphere/
    # Where your training set lives
    training_dir: ../../xdisk/training_set_eft_2t_2z_hypersphere/
    # Location of config file with cosmology + bias parameters
    # This file contains all of the parameter ranges
    cosmo_dir : ./configs/cosmo_pars/cosmo_pars_2t_2z.yaml

    model_type : stacked_transformer
    loss_type : hyperbolic_chi2

    num_cosmo_params    : 5
    num_nuisance_params : 3 # <- per tracer

    # specifications are for each network - will be repeated for each sample / redshift bin
    galaxy_ps_emulator:
        # mlp parameters
        spectrum_embed_dim  : 8
        num_mlp_blocks      : 2
        num_block_layers    : 5
        use_skip_connection : True

        # transformer parameters
        num_transformer_blocks : 2
        split_dim              : 5
        split_size             : 20

    # Training parameters
    num_epochs: 500
    galaxy_ps_learning_rate: 0.005
    batch_size: 1000
    training_set_fraction : 1.0
    early_stopping_epochs: 25
    weight_initialization: He
    optimizer_type : Adam

    # whether to re-calculate the training-set loss at the end of each epoch
    # Setting to true gives more accurate loss stats, but is slower
    # Validation set loss is not changed by this option
    recalculate_train_loss : False
    # whether to attempt training on GPU
    # NOTE: Small networks will possibly train faster on CPU!
    use_gpu : True

Here are some important considerations to make before training:

- You'll need to decide whether to try training your network on a CPU or GPU (assuming one is available to you). In general, networks with transformers train **significantly** faster on  GPUs, so we recommend you try training on GPUs whenever possible. By defualt, mentat-lss will attempt to use a GPU if it is available.
- The specific binning information (num_zbins, num_ells, etc) must match those found in the training set. The code will throw an error on startup if they don't.
- The above specifications are the optimized values found in PAPER_LINK_HERE. The optimal setup will potentially be different for your case, but these values should provide a good starting point.

Once you have your configs all sorted. You can train you network using the following script,

.. code-block:: python

    from mentat_lss.emulator import pk_emulator
    import mentat_lss.training_loops as training_loops
    import logging

    config_file = "/path/to/config_file.yaml"

    # Used for printing output during training. If you don't want any
    # output. set to logging.WARNING
    logging.basicConfig(level = logging.INFO)

    t1 = time.time()
    emulator = pk_emulator(config_file, "train")

    # train on a single cpu / gpu
    training_loops.train_on_single_device(emulator)

We have also provided a more robust script in `scripts/train_emulator.py` that also
handles training on multiple GPUs. Fore more details, see :doc:`tutorials/training`.

During the actual training process, `mentat-lss` will loop through each subnet, each of which
correspond to a single tracer / redshift bin. It will then print out the average training set and 
validation set loss values, as well as the number of epochs elapses since the validation loss improved.::

    `Net idx : [ps, z], epoch: N, avg train loss: l1, avg validation loss: l2 (epochs_since_improved)`

This will repeat until either the validation loss for all sub-nets hasn't improved for 25 epochs, or if max_epochs is reached.

Testing the Emulator
--------------------

We provide an example jupyter notebook for running various tests on your emulator :doc:`here <tutorials/test_emulator>`.

Using the Emulator
-------------------

Finally, once you are sure your emulator works, you can generate power spectrum with,

.. code-block:: python
    
    emulator = ps_emulator(emu_dir, "eval")
    pk_predict = ps_emulator.get_power_spectra(input_params)

which will output power spectrum multipoles as a numpy array with shape 
``[nps, nz, nk, nl]``. You can access the required input parameters (and their order) with,

.. code-block:: python

    # REQUIRED: input_params should be in the same order as emu_params
    emu_params = ps_emulator.get_required_emu_params()
    # OPTIONAL: should be concatenated at the end of emu_params
    analytic_params = ps_emulator.get_required_analytic_parameters()

Note that if you include analytic_params, the code will compute the EFT counterterms and
stochastic contributions. Under the hood, the code calls symbolic_pofk_
to quickly calculate the linear matter power spectrum.

.. _symbolic_pofk: https://github.com/DeaglanBartlett/symbolic_pofk

You can then hook up this method to your favorite MCMC sampler
to run some likelihood analyses!

