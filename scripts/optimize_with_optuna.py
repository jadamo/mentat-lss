import os
import logging
import optuna
import shutil
import yaml
import argparse
import torch 

import mentat_lss.training_loops as training_loops
from mentat_lss.utils import load_config_file, calc_chi2_statistics
from mentat_lss.emulator import ps_emulator

def create_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

def clean_cache(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def define_model(trial, cache_dir, default_config_file, training_set_fraction, device=None):

    trial_config_file = default_config_file.copy()

    num_mlp_blocks = trial.suggest_int("num_mlp_blocks", 1, 6)
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 0, 1)
    num_block_layers = trial.suggest_int("num_block_layers", 2, 6)
    hidden_dim_factor = trial.suggest_float("hidden_dim_factor", 1.0, 2.0)
    batch_size = trial.suggest_int("batch_size", 100, 1000, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    # token_proj_dim is a multiple of 8, so num_heads in [1,2,4,8] always divides it.
    token_proj_dim = trial.suggest_int("token_proj_dim", 8, 64, step=8)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    spectrum_embed_dim = trial.suggest_int("spectrum_embed_dim", 2, 10)

    trial_config_file["galaxy_ps_emulator"]["num_mlp_blocks"] = num_mlp_blocks
    trial_config_file["galaxy_ps_emulator"]["num_block_layers"] = num_block_layers
    trial_config_file["galaxy_ps_emulator"]["hidden_dim_factor"] = hidden_dim_factor
    trial_config_file["galaxy_ps_emulator"]["num_transformer_blocks"] = num_transformer_blocks
    trial_config_file["galaxy_ps_emulator"]["token_proj_dim"] = token_proj_dim
    trial_config_file["galaxy_ps_emulator"]["spectrum_embed_dim"] = spectrum_embed_dim
    trial_config_file["galaxy_ps_emulator"]["num_heads"] = num_heads
    trial_config_file["batch_size"] = batch_size
    trial_config_file["galaxy_ps_learning_rate"] = learning_rate
    # to save time, only train with 10% of the full data and for only 150 epochs
    trial_config_file["training_set_fraction"] = training_set_fraction
    trial_config_file["num_epochs"] = 300

    # Use an absolute path for save_dir so os.path.join(input_dir, save_dir) resolves
    # to this location regardless of what input_dir is set to.
    trial_config_file["save_dir"] = os.path.join(cache_dir, f"trial_{trial.number}")
    file_path = os.path.join(cache_dir, f'config_{trial.number}.yaml')
    with open(file_path, 'w') as outfile:
        yaml.dump(dict(trial_config_file), outfile, sort_keys=False, default_flow_style=False)

    return ps_emulator(file_path, "train", device=device)

def save_best_params(save_loc, default_config_file, best_params):
    best_config_file = default_config_file.copy()

    best_config_file["galaxy_ps_emulator"]["num_mlp_blocks"] = best_params["num_mlp_blocks"]
    best_config_file["galaxy_ps_emulator"]["num_block_layers"] = best_params["num_block_layers"]
    best_config_file["galaxy_ps_emulator"]["hidden_dim_factor"] = best_params["hidden_dim_factor"]
    best_config_file["galaxy_ps_emulator"]["num_transformer_blocks"] = best_params["num_transformer_blocks"]
    best_config_file["galaxy_ps_emulator"]["token_proj_dim"] = best_params["token_proj_dim"]
    best_config_file["galaxy_ps_emulator"]["spectrum_embed_dim"] = best_params["spectrum_embed_dim"]
    best_config_file["galaxy_ps_emulator"]["num_heads"] = best_params["num_heads"]
    best_config_file["batch_size"] = best_params["batch_size"]
    best_config_file["galaxy_ps_learning_rate"] = best_params["learning_rate"]

    dir_name = os.path.dirname(save_loc)
    base_name = os.path.splitext(os.path.basename(save_loc))[0]
    file_path = os.path.join(dir_name, f'{base_name}_optimized.yaml')
    print(f"Saving optimized hyperparameters to {file_path}...")
    with open(file_path, 'w') as outfile:
        yaml.dump(dict(best_config_file), outfile, sort_keys=False, default_flow_style=False)

def objective(trial, cache_dir, default_config_file, training_set_fraction, device=None):
    emulator = define_model(trial, cache_dir, default_config_file, training_set_fraction, device)

    training_loops.train_on_single_device(emulator, trial)

    # Use the median delta chi2 from the full emulator validation set as the objective
    valid_loader = emulator.load_data("validation")
    chi2_stats = calc_chi2_statistics(emulator, valid_loader, calc_partial=False)
    result = torch.median(chi2_stats).item()

    # clean up trial save directory to avoid filling disk
    trial_save_dir = os.path.join(emulator.input_dir, emulator.save_dir)
    if os.path.exists(trial_save_dir):
        shutil.rmtree(trial_save_dir)

    return result

def run_worker(gpu_id, cache_dir, default_config_file, n_trials, db_path, training_set_fraction):

    device = torch.device(f"cuda:{gpu_id}")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}}  # wait up to 30s for lock
    )
    study = optuna.load_study(
        study_name="combined_tracer",
        storage=storage
    )
    study.optimize(
        lambda trial: objective(trial, cache_dir, default_config_file, training_set_fraction, device),
        n_trials=n_trials,
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="<Required> Path to config file."
    )
    parser.add_argument(
        "n_trials", type=int, default=None,
        help="<Required> number of trials to execute by Optuna."
    )
    parser.add_argument(
        "training_set_fraction", type=float, default=0.1,
        help="Fraction of the full training set to use for each trial. Set to a smaller value to speed up trials at the cost of more noise in the objective. Defaults to 0.1."
    )
    parser.add_argument(
        "db_path", type=str, default="optuna_results.db",
        help="Path to Optuna SQLite DB. Must be on a shared filesystem for multi-node."
    )
    parser.add_argument(
        "cache_dir", type=str, default="./optuna_cache",
        help="Base directory for per-node trial caches. Each node writes to <cache-dir>/node<N>/."
    )
    command_line_args = parser.parse_args()

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger("")
    try:
        node_id = int(os.environ.get("SLURM_NODEID", 0))
        num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    except (ValueError, TypeError):
        node_id = 0
        num_nodes = 1
        logger.warning("Failed to parse SLURM_NODEID, defaulting to 0")
    num_gpus = torch.cuda.device_count() # <- GPUs on the given node
    db_path = command_line_args.db_path
    training_set_fraction = command_line_args.training_set_fraction

    # Create study once (workers load it)
    try:
        optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///{db_path}",
            study_name="combined_tracer",
            pruner=optuna.pruners.MedianPruner(),
        )
    except optuna.exceptions.DuplicatedStudyError:
        pass  # another node created the study first

    cache_dir = os.path.abspath(os.path.join(command_line_args.cache_dir, f"node{node_id}"))
    if node_id == 0:
        logger.info(f"Running {command_line_args.n_trials} trials using {num_gpus} GPUs across {num_nodes} nodes for optimization.")
        logger.info(f"Optuna results will be saved to {db_path}")
        logger.info(f"Each trial will use {training_set_fraction*100}% of the full training data and last a max of 300 epochs.")
    
    logger.info(f"Node {node_id} creating cache directory {cache_dir}...")
    create_cache(cache_dir)

    default_config_file = load_config_file(command_line_args.config_file)

    if num_gpus < 2:
        # Single GPU / CPU fallback
        device = torch.device("cuda:0") if num_gpus == 1 else torch.device("cpu")
        logger.info(f"Running trials on device {device}")
        study = optuna.load_study(study_name="combined_tracer", storage=f"sqlite:///{db_path}")
        study.optimize(
            lambda trial: objective(trial, cache_dir, default_config_file, training_set_fraction, device),
            n_trials=command_line_args.n_trials,
        )
    else:
        import torch.multiprocessing as mp
        logger.info(f"Running trials on {num_gpus} GPUs with multiprocessing...")
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            run_worker,
            args=(cache_dir, default_config_file, command_line_args.n_trials // num_gpus // num_nodes, db_path, training_set_fraction),
            nprocs=num_gpus,
            join=True
        )
    study = optuna.load_study(study_name="combined_tracer", storage=f"sqlite:///{db_path}")

    if node_id == 0:
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        save_best_params(command_line_args.config_file, default_config_file, trial.params)
    
    clean_cache(cache_dir)

if __name__ == "__main__":
    main()
