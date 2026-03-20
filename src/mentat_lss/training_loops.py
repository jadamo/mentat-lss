import torch
import time
import itertools
import logging
import os

import optuna
from mentat_lss.emulator import ps_emulator, cov_emulator, compile_multiple_device_training_results
from mentat_lss.utils import calc_avg_ps_loss, calc_avg_cov_loss, normalize_cosmo_params


def train_galaxy_ps_one_epoch(emulator:ps_emulator, train_loader:torch.utils.data.DataLoader, bin_idx:list):
    """Runs through one epoch of training for one sub-network in the galaxy_ps model

    Args:
        emulator (ps_emulator): emulator object to train
        train_loader (torch.utils.data.DataLoader): training data to loop through
        bin_idx (list): bin index [ps, z] or [z] identifying the sub-network to train.

    Returns:
        avg_loss (torch.Tensor): Average training-set loss. Used for backwards propagation
    """
    total_loss = 0.
    total_time = 0
    if emulator.model_type == "combined_tracer_transformer":
        net_idx  = bin_idx
        is_cross = net_idx >= emulator.num_zbins
        z_idx    = net_idx - emulator.num_zbins if is_cross else net_idx
    else:
        ps_idx = bin_idx[0]
        z_idx  = bin_idx[1]
        net_idx = (z_idx * emulator.num_spectra) + ps_idx
    for (i, batch) in enumerate(train_loader):
        t1 = time.time()

        # setup input parameters
        params = normalize_cosmo_params(batch[0], emulator.input_normalizations)
        params = emulator.galaxy_ps_model.organize_parameters(params)

        prediction = emulator.galaxy_ps_model.forward(params, net_idx)

        if emulator.model_type == "combined_tracer_transformer":
            spec_indices = emulator.galaxy_ps_model.cross_spectrum_indices if is_cross \
                           else emulator.galaxy_ps_model.auto_spectrum_indices
            target = torch.flatten(batch[1][:, spec_indices, z_idx], start_dim=0, end_dim=1)
        else:
            target = torch.flatten(batch[1][:,ps_idx,z_idx], start_dim=1)

        # calculate loss and update network parameters
        loss = emulator.loss_function(prediction, target, emulator.invcov_full, True)
        assert torch.isnan(loss) == False
        assert torch.isinf(loss) == False
        if emulator.model_type == "combined_tracer_transformer":
            emulator.optimizer[net_idx].zero_grad(set_to_none=True)
            loss.backward()
            emulator.optimizer[net_idx].step()
        else:
            emulator.optimizer[ps_idx][z_idx].zero_grad(set_to_none=True)
            loss.backward()
            emulator.optimizer[ps_idx][z_idx].step()

        total_loss += loss.detach()
        total_time += (time.time() - t1)

    emulator.logger.debug("time for epoch: {:0.1f}s, time per batch: {:0.1f}ms".format(total_time, 1000*total_time / len(train_loader)))
    return (total_loss / len(train_loader.dataset))


def train_on_single_device(emulator:ps_emulator, trial=None):
    """Trains the emulator on a single device (cpu or gpu)

    Args:
        emulator (ps_emulator): network object to train.
        trial (optuna.trial.Trial, optional): If not None, the current trial informaiton
            from optuna. Default None
    """

    # load training / validation datasets
    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    if emulator.model_type == "combined_tracer_transformer":
        bin_idx_list = list(range(2 * emulator.num_zbins))
        total_num_nets = 2 * emulator.num_zbins
    else:
        bin_idx_list = list(itertools.product(range(emulator.num_spectra), range(emulator.num_zbins)))
        total_num_nets = emulator.num_spectra * emulator.num_zbins

    best_loss           = [torch.inf for i in range(total_num_nets)]
    epochs_since_update = [0 for i in range(total_num_nets)]
    emulator._init_training_stats()
    emulator._init_optimizer()
    emulator.galaxy_ps_model.train()

    start_time = time.time()
    # loop thru epochs
    for epoch in range(emulator.num_epochs):

        # loop thru individual networks
        for bin_idx in bin_idx_list:
            if emulator.model_type == "combined_tracer_transformer":
                net_idx  = bin_idx
                is_cross = net_idx >= emulator.num_zbins
                z        = net_idx - emulator.num_zbins if is_cross else net_idx
                net_id_str = f"{'cross' if is_cross else 'auto'}[{z}]"
            else:
                ps = bin_idx[0]
                z = bin_idx[1]
                net_idx = (z * emulator.num_spectra) + ps
                net_id_str = f"[{ps}, {z}]"
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                continue

            training_loss = train_galaxy_ps_one_epoch(emulator, train_loader, bin_idx)
            if emulator.recalculate_train_loss:
                emulator.train_loss[net_idx].append(calc_avg_ps_loss(emulator, train_loader, emulator.loss_function, bin_idx))
            else:
                emulator.train_loss[net_idx].append(training_loss)
            emulator.valid_loss[net_idx].append(calc_avg_ps_loss(emulator, valid_loader, emulator.loss_function, bin_idx))


            emulator.scheduler[net_idx].step(emulator.valid_loss[net_idx][-1])
            emulator.train_time = time.time() - start_time

            if emulator.valid_loss[net_idx][-1] < best_loss[net_idx]:
                best_loss[net_idx] = emulator.valid_loss[net_idx][-1]
                epochs_since_update[net_idx] = 0
                emulator._update_checkpoint(net_idx, "galaxy_ps")
            else:
                epochs_since_update[net_idx] += 1

            emulator.logger.info(f"Net idx : {net_id_str}, Epoch : {epoch}, avg train loss: {emulator.train_loss[net_idx][-1]:0.4e}\t avg validation loss: {emulator.valid_loss[net_idx][-1]:0.4e}\t ({epochs_since_update[net_idx]})")

            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                emulator.logger.info(f"Model {net_id_str} has not improved for {epochs_since_update[net_idx]} epochs. Initiating early stopping...")

        if trial != None:
            accuracy = torch.mean([emulator.valid_loss[net_idx][-1] for net_idx in range(len(emulator.valid_loss))])
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

def train_on_multiple_devices(gpu_id:int, net_indeces:list, config_dir:str):
    """Trains the given network on multiple gpu devices by splitting.

    This function is called in parralel using multiproccesing, and works by training specific sub-networks 
    on seperate gpus, each saving to a seperate sub-directory. After 25 epochs have passed on gpu 0, the results from all gpus are compiles together
    and saved in the base save directory

    Args:
        gpu_id (int): gpu number for logging and organizing save location.
        net_indeces (list): List of sub-network indices to train on the given gpu. This is different for each gpu
        config_dir (str): Location of the input network config file.
    """
    # Each sub-process gets its own indpendent emulator object, where it will train the corresponding
    # sub-networks based on net_indeces
    device = torch.device(f"cuda:{gpu_id}")
    logging.basicConfig(level=logging.DEBUG, format=f"[GPU {gpu_id}] %(message)s")
    emulator = ps_emulator(config_dir, "train", device)

    base_save_dir = os.path.join(emulator.input_dir, emulator.save_dir)
    emulator.save_dir += "rank_"+str(gpu_id)+"/"
    emulator.logger.debug(f"training networks with ids: {net_indeces[gpu_id]}")

    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    emulator._init_training_stats()
    num_nets = len(emulator.train_loss)
    best_loss           = [torch.inf for i in range(num_nets)]
    epochs_since_update = [0 for i in range(num_nets)]
    emulator._init_optimizer()

    emulator.galaxy_ps_model.train()

    start_time = time.time()
    # loop thru epochs
    for epoch in range(emulator.num_epochs):
        # loop thru individual networks
        for bin_idx in net_indeces[gpu_id]:
            if emulator.model_type == "combined_tracer_transformer":
                net_idx  = bin_idx
                is_cross = net_idx >= emulator.num_zbins
                z        = net_idx - emulator.num_zbins if is_cross else net_idx
                net_id_str = f"{'cross' if is_cross else 'auto'}[{z}]"
            else:
                ps = bin_idx[0]
                z = bin_idx[1]
                net_idx = (z * emulator.num_spectra) + ps
                net_id_str = f"[{ps}, {z}]"
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                continue

            training_loss = train_galaxy_ps_one_epoch(emulator, train_loader, bin_idx)
            if emulator.recalculate_train_loss:
                emulator.train_loss[net_idx].append(calc_avg_ps_loss(emulator, train_loader, emulator.loss_function, bin_idx))
            else:
                emulator.train_loss[net_idx].append(training_loss)
            emulator.valid_loss[net_idx].append(calc_avg_ps_loss(emulator, valid_loader, emulator.loss_function, bin_idx))


            emulator.scheduler[net_idx].step(emulator.valid_loss[net_idx][-1])
            emulator.train_time = time.time() - start_time

            if emulator.valid_loss[net_idx][-1] < best_loss[net_idx]:
                best_loss[net_idx] = emulator.valid_loss[net_idx][-1]
                epochs_since_update[net_idx] = 0
                emulator._update_checkpoint(net_idx, "galaxy_ps")
            else:
                epochs_since_update[net_idx] += 1

            emulator.logger.info(f"Net idx : {net_id_str}, Epoch : {epoch}, avg train loss: {emulator.train_loss[net_idx][-1]:0.4e}\t avg validation loss: {emulator.valid_loss[net_idx][-1]:0.4e}\t ({epochs_since_update[net_idx]})")
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                emulator.logger.info(f"Model {net_id_str} has not improved for {epochs_since_update[net_idx]} epochs. Initiating early stopping...")

        if gpu_id == 0 and epoch % 5 == 0 and epoch > 0:
            emulator.logger.info("Checkpointing progress from all devices...")
            full_emulator = compile_multiple_device_training_results(base_save_dir, config_dir, emulator.num_gpus)
            full_emulator._save_model()


def train_cov_one_epoch(emulator:cov_emulator, train_loader:torch.utils.data.DataLoader, z_idx:int):
    """Runs through one epoch of training for one redshift bin of the covariance emulator.

    Args:
        emulator (cov_emulator): emulator object to train
        train_loader (torch.utils.data.DataLoader): training data to loop through
        z_idx (int): index of the redshift bin sub-network to train

    Returns:
        avg_loss (float): average training-set L1 loss for this epoch
    """
    from torch.nn import functional as F

    total_loss = 0.
    total_time = 0.
    emulator.cov_model.train()

    for params, matrices in train_loader:
        t1 = time.time()

        org_params  = emulator.cov_model.organize_parameters(params)
        norm_params = normalize_cosmo_params(org_params, emulator.input_normalizations)
        prediction  = emulator.cov_model(norm_params, z_idx=z_idx)
        target      = matrices[:, z_idx]

        loss = F.l1_loss(prediction, target, reduction="sum")

        assert torch.isnan(loss) == False
        assert torch.isinf(loss) == False

        emulator.optimizer[z_idx].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(emulator.cov_model.networks[z_idx].parameters(), 1e8)
        emulator.optimizer[z_idx].step()

        total_loss += loss.detach()
        total_time += time.time() - t1

    emulator.logger.debug("z={:d}, time for epoch: {:.1f}s, time per batch: {:.1f}ms".format(
        z_idx, total_time, 1000 * total_time / len(train_loader)))
    return (total_loss / len(train_loader.dataset)).item()


def _calc_avg_cov_loss(emulator:cov_emulator, data_loader:torch.utils.data.DataLoader, z_idx:int):
    """Calculates the average L1 loss for one redshift bin over the given dataset.

    Args:
        emulator (cov_emulator): emulator object to evaluate
        data_loader (torch.utils.data.DataLoader): dataset to evaluate on
        z_idx (int): index of the redshift bin to evaluate

    Returns:
        avg_loss (float): average L1 loss per sample
    """
    from torch.nn import functional as F

    emulator.cov_model.eval()
    avg_loss = 0.
    with torch.no_grad():
        for params, matrices in data_loader:
            org_params  = emulator.cov_model.organize_parameters(params)
            norm_params = normalize_cosmo_params(org_params, emulator.input_normalizations)
            prediction  = emulator.cov_model(norm_params, z_idx=z_idx)
            target      = matrices[:, z_idx]
            avg_loss += F.l1_loss(prediction, target, reduction="sum").item()
    return avg_loss / len(data_loader.dataset)


def train_cov_on_single_device(emulator:cov_emulator):
    """Trains the covariance emulator on a single device (CPU or GPU).

    Trains each redshift-bin sub-network independently with early stopping
    per zbin. The best checkpoint per zbin (by validation loss) is saved to
    disk via emulator._update_checkpoint(z_idx).

    Args:
        emulator (cov_emulator): emulator object to train
    """
    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    best_loss           = [torch.inf] * emulator.num_zbins
    epochs_since_update = [0]         * emulator.num_zbins

    emulator._init_training_stats()
    emulator._init_optimizer()
    emulator.cov_model.train()

    start_time = time.time()

    for epoch in range(emulator.num_epochs):
        for z in range(emulator.num_zbins):
            if epochs_since_update[z] > emulator.early_stopping_epochs:
                continue

            train_loss = train_cov_one_epoch(emulator, train_loader, z)
            valid_loss = _calc_avg_cov_loss(emulator, valid_loader, z)

            emulator.train_loss[z].append(train_loss)
            emulator.valid_loss[z].append(valid_loss)
            emulator.train_time = time.time() - start_time

            emulator.scheduler[z].step(valid_loss)

            if valid_loss < best_loss[z]:
                best_loss[z] = valid_loss
                epochs_since_update[z] = 0
                emulator._update_checkpoint(z)
            else:
                epochs_since_update[z] += 1

            emulator.logger.info(
                "z={:d}, Epoch {:d}, avg train loss: {:.4e}, avg valid loss: {:.4e} ({:d})".format(
                    z, epoch, train_loss, valid_loss, epochs_since_update[z]))

            if emulator.early_stopping_epochs != -1 and \
               epochs_since_update[z] >= emulator.early_stopping_epochs:
                emulator.logger.info(
                    f"z={z}: validation loss has not improved for "
                    f"{epochs_since_update[z]} epochs. Initiating early stopping.")
