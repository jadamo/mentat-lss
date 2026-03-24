import itertools

import torch
import os, math
import pytest
import itertools

import mentat_lss.emulator as emulator
from mentat_lss.models.blocks import *

def test_linear_with_channels():
    # test that the linear_with_channels sub-block treats channels independently

    parallel_layers = linear_with_channels(10, 10, 2)
    with torch.no_grad():
        parallel_layers.w[0,:,:] = 1.
        parallel_layers.b[0,:,:] = 0.

    for n in range(100):
        test_input = torch.rand((1, 2, 10))
        test_output = parallel_layers(test_input)
        
        assert torch.all(test_output[:,1] != torch.sum(test_input[:,1]))

@pytest.mark.parametrize("input_dim, output_dim, num_layers, hidden_dim_factor, expected", [
    (10, 10, 2, 1., None), 
    (1, 10, 3, 1.5, None),
    (1, 10, 3, 0.9, None),
    (0, 10, 3, 1.0, ValueError),
    (10, 0, 3, 1.0, ValueError),
    (10, 10, 0, 1.0, ValueError),
])
def test_block_resnet(input_dim, output_dim, num_layers, hidden_dim_factor, expected):

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            resnet_block = block_resnet(input_dim, output_dim, hidden_dim_factor, num_layers, skip_connection=True)
    else:
        test_input = torch.rand((100, input_dim))
        resnet_block = block_resnet(input_dim, output_dim, hidden_dim_factor, num_layers, skip_connection=True)
        test_output = resnet_block(test_input)

        assert test_output.shape == (100, output_dim)
        assert not torch.all(torch.isnan(test_output))
        assert not torch.all(torch.isinf(test_output))

@pytest.mark.parametrize("embedding_dim, split_dim, expected", [
    (10, 10, None),
    (10, 5, None), 
    (2, 10, ValueError),
    (0, 10, ValueError),
    (10, 4, ValueError),
    (10, 0, ValueError),
])
def test_transformer_block(embedding_dim, split_dim, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            transformer_block = block_transformer_encoder(embedding_dim, split_dim, 0.1)
    else:
        transformer_block = block_transformer_encoder(embedding_dim, split_dim, 0.1)
        test_input = torch.rand(100, embedding_dim)
        test_output = transformer_block(test_input)

        assert test_output.shape == (100, embedding_dim)
        assert not torch.all(torch.isnan(test_output))
        assert not torch.all(torch.isinf(test_output))

@pytest.mark.parametrize("model_type", [
    "stacked_transformer",
    "combined_tracer_transformer",
])
def test_network_forward(model_type):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(current_dir, "test_configs", f"network_pars_{model_type}.yaml")

    test_emulator = emulator.ps_emulator(test_dir, "train")

    param_size = test_emulator.num_cosmo_params + \
                 (test_emulator.num_nuisance_params * test_emulator.num_zbins * test_emulator.num_tracers)
    test_input   = torch.randn(2, param_size, device=test_emulator.device)
    test_input_2 = torch.randn(2, param_size, device=test_emulator.device)

    test_emulator.galaxy_ps_model.eval()
    test_input   = test_emulator.galaxy_ps_model.organize_parameters(test_input)
    test_input_2 = test_emulator.galaxy_ps_model.organize_parameters(test_input_2)

    test_output_sub    = test_emulator.galaxy_ps_model.forward(test_input, 0)
    test_output_full   = test_emulator.galaxy_ps_model.forward(test_input)
    test_output_full_2 = test_emulator.galaxy_ps_model.forward(test_input_2)

    output_dim = test_emulator.num_kbins * test_emulator.num_ells

    assert test_output_full.shape == (2, test_emulator.num_spectra, test_emulator.num_zbins, output_dim)
    assert not torch.all(torch.isnan(test_output_full))
    assert not torch.all(torch.isinf(test_output_full))
    assert not torch.allclose(test_output_full, test_output_full_2)

    if model_type == "stacked_transformer":
        assert test_output_sub.shape == (2, output_dim)
        assert torch.allclose(test_output_sub, test_output_full[:, 0, 0])
    else:
        assert test_output_sub.shape == (2 * test_emulator.num_tracers, output_dim)

def test_activation_functions():

    # test that the activation functions do not produce NaNs or infs for typical inputs
    test_input = torch.randn(100)

    func = activation_function(100)
    gelu_output = func(test_input)
    assert not torch.all(torch.isnan(gelu_output))
    assert not torch.all(torch.isinf(gelu_output))

def test_combined_tracer_transformer_organize_params():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(current_dir, "test_configs", "network_pars_combined_tracer_transformer.yaml")

    # constructes the network
    test_emulator = emulator.ps_emulator(test_dir, "train")

    # generate an input sequence and pass it through the network
    # test_input = torch.Tensor([list(range(test_emulator.num_cosmo_params)) + \
    #                            list(range(test_emulator.num_cosmo_params, test_emulator.num_cosmo_params + \
    #                            test_emulator.num_nuisance_params * test_emulator.num_zbins * test_emulator.num_tracers))]).to(test_emulator.device)
    test_input = torch.randn((10, test_emulator.num_cosmo_params + \
                                 (test_emulator.num_nuisance_params *test_emulator.num_zbins * test_emulator.num_tracers)),
                                 device = test_emulator.device)

    organized_input = test_emulator.galaxy_ps_model.organize_parameters(test_input)
    assert organized_input[0].shape == (10 *test_emulator.num_tracers, test_emulator.num_zbins,
                                       test_emulator.num_cosmo_params + test_emulator.num_nuisance_params)
    assert organized_input[1].shape == (10 *math.comb(test_emulator.num_tracers, 2), test_emulator.num_zbins,
                                       test_emulator.num_cosmo_params + 2*test_emulator.num_nuisance_params)

    for b in range(test_input.shape[0]):
        for z in range(test_emulator.num_zbins):
            # stride matching the reshape layout: [nn, nz, nt] → stride per nuisance param = nz*nt
            iterate = test_emulator.num_zbins * test_emulator.num_tracers
            auto_iter = 0
            cross_iter = 0
            for s1, s2 in itertools.product(range(test_emulator.num_tracers), repeat=2):
                if s1 > s2: continue
                if s1 == s2:
                    idx1 = z*test_emulator.num_tracers + s1
                    out_idx = b * test_emulator.num_tracers + auto_iter
                    assert torch.all(organized_input[0][out_idx,z,:test_emulator.num_cosmo_params] == test_input[b, :test_emulator.num_cosmo_params])
                    assert torch.all(organized_input[0][out_idx,z,test_emulator.num_cosmo_params:test_emulator.num_cosmo_params + test_emulator.num_nuisance_params] == \
                                                        test_input[b, test_emulator.num_cosmo_params + idx1::iterate])
                    auto_iter += 1
                else:
                    idx1 = z*test_emulator.num_tracers + s1
                    idx2 = z*test_emulator.num_tracers + s2
                    out_idx = b * math.comb(test_emulator.num_tracers, 2) + cross_iter
                    assert torch.all(organized_input[1][out_idx,z,:test_emulator.num_cosmo_params] == test_input[b, :test_emulator.num_cosmo_params])
                    assert torch.all(organized_input[1][out_idx,z,test_emulator.num_cosmo_params:test_emulator.num_cosmo_params + 2*test_emulator.num_nuisance_params] == \
                                                        torch.concatenate([test_input[b, test_emulator.num_cosmo_params + idx1::iterate],
                                                                           test_input[b, test_emulator.num_cosmo_params + idx2::iterate]]))
                    cross_iter += 1


@pytest.mark.parametrize("model_mode, expected", [
    ("train", None),
    ("eval", None),
    ("invalid_mode", KeyError)
])
def test_emulator_mode(model_mode, expected):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(current_dir, "test_configs", "network_pars_stacked_transformer.yaml")
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            test_emulator = emulator.ps_emulator(test_dir, model_mode)
    else:

        # need to train and save a model first to test loading in eval mode
        if model_mode == "eval":
            train_emulator = emulator.ps_emulator(test_dir, "train")
            train_emulator._init_training_stats()
            train_emulator._load_ps_properties(os.path.join(current_dir, "test_configs"))
            train_emulator._save_model()
            test_dir = os.path.join(current_dir, "test_networks", "stacked_transformer")

        test_emulator = emulator.ps_emulator(test_dir, model_mode)
        assert test_emulator.galaxy_ps_model is not None

@pytest.mark.parametrize("model_type,", [
    ("stacked_transformer"),
    ("combined_tracer_transformer")
])
def test_save_and_load(model_type):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_config_file = os.path.join(current_dir, "test_configs", f"network_pars_{model_type}.yaml")
    test_dir = os.path.join(current_dir, "test_networks", f"{model_type}")
    save_dir = os.path.join(current_dir, "test_networks", f"{model_type}")

    # constructes the network
    test_emulator = emulator.ps_emulator(test_config_file, "train", device="cpu")
    test_emulator._init_training_stats()
    test_emulator._load_ps_properties(os.path.join(current_dir, "test_configs"))
    initial_dict = test_emulator.galaxy_ps_model.state_dict()

    test_emulator._save_model()
    loaded_emulator = emulator.ps_emulator(test_dir, "eval", device="cpu")
    loaded_dict = loaded_emulator.galaxy_ps_model.state_dict()
    # check that the state_dicts are the same

    for key in initial_dict.keys():
        assert torch.all(initial_dict[key] == loaded_dict[key])

    if os.path.exists(save_dir):
        os.system(f"rm -r {save_dir}")

def test_get_power_spectra():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(current_dir, "test_configs", "network_pars_stacked_transformer.yaml")

    # constructes the network
    test_emulator = emulator.ps_emulator(test_dir, "train")

    # generate a random input sequence and pass it through the network
    test_input = torch.randn(1, test_emulator.num_cosmo_params + \
                                (test_emulator.num_nuisance_params *test_emulator.num_zbins * test_emulator.num_tracers),
                                device = test_emulator.device)
    test_emulator.galaxy_ps_model.eval()
    test_output = test_emulator.get_power_spectra(test_input)

    assert test_output.shape == (1, test_emulator.num_spectra,
                                 test_emulator.num_zbins,
                                 test_emulator.num_kbins,
                                 test_emulator.num_ells)
    assert np.all(np.isnan(test_output)) == False
    assert np.all(np.isinf(test_output)) == False