"""Script for model evaluation and output generation."""

import argparse
import json
import os

import h5py
import imageio
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from dataset import get_dataloader
from model import PointSetNet
from train import step, visualize
from utils import toy_render

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--set', nargs='+')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_outputs(pos, grp, section_id, save_dir, zlim):
    """
    Create video visualization of outputs.

    Input:
        pos: tensor of shape (t, N, 3)
        grp: tensor of shape (t, N, G)
        section_id: int
        save_dir: str
    """
    n_particles = pos.shape[1]
    highlight_idxs = np.random.choice(range(n_particles), 5, replace=False)
    vis_frames = toy_render(
        pos, title='Output', highlight_idxs=highlight_idxs, groups=grp,
        zlim=zlim)

    imageio.mimwrite(
        os.path.join(save_dir, 'output_{}.mp4'.format(section_id)),
        vis_frames,
        fps=5)


def generate_outputs(config, model, n_traj, images, save_dir):
    """
    Generate perception module outputs for inference module.

    For each (T, C, H, W) input sequence, generate outputs:
    - positions (T - t + 1) x (t, N, 3)
    - groups (T - t + 1) x (t, N, G)
    where t = config.n_frames_eval (NOT config.n_frames)
    """
    model.eval()

    B, T, C, H, W = images.shape

    # use special z-axis range for MassRope visualization
    if config.dataset == 'MassRope':
        zlim = (0.3, 1.7)
    else:
        zlim = (0, 0.7)

    if config.vis_eval:
        vis_save_dir = os.path.join(save_dir, 'out_vis')

    if config.n_frames >= config.n_frames_eval:
        # use model outputs directly

        n_sections = T - config.n_frames_eval + 1

        # sliding window rollout, taking all n_frames in each output
        for j in range(n_sections):
            # pred_pos (B, n_frames, 3, N) -> (B, n_frames, N, 3)
            # pred_grp (B, n_frames, N, G)
            pred_pos, pred_grp = model(
                images[:, j:j+config.n_frames, ...].to(_DEVICE))
            pred_pos = pred_pos.permute(0, 1, 3, 2)

            if config.vis_eval and j % config.vis_eval_every == 0:
                # visualize output
                vis_pos = pred_pos[
                    0, j:j+config.n_frames_eval, ...].detach().cpu().numpy()
                vis_grp = pred_grp[
                    0, j:j+config.n_frames_eval, ...].detach().cpu().numpy()
                visualize_outputs(vis_pos, vis_grp, j, vis_save_dir, zlim)

            for i in range(B):
                # create save directory for all the data in this batch
                data_id = n_traj + i
                data_dir = os.path.join(save_dir, str(data_id))
                if not os.path.isdir(data_dir):
                    os.makedirs(data_dir)

                t_pos = pred_pos[
                    i, j:j+config.n_frames_eval,
                    ...].detach().cpu().numpy()  # (t, N, 3)
                t_grp = pred_grp[
                    i, j:j+config.n_frames_eval,
                    ...].detach().cpu().numpy()  # (t, N, G)

                h5f = h5py.File(os.path.join(data_dir, str(j) + '.h5'), 'w')
                h5f.create_dataset('positions', data=t_pos)
                h5f.create_dataset('groups', data=t_grp)

    else:
        # config.n_frames < config.n_frames_eval
        # take model outputs one frame at a time except for the first rollout

        n_sections = T - config.n_frames_eval + 1

        # sliding window rollout, taking only the last frame in each output
        # (except for the first output; all frames are taken)
        last_frames_pos = []
        last_frames_grp = []

        for j in range(T - config.n_frames + 1):
            # pred_pos (B, n_frames, 3, N) -> (B, n_frames, N, 3)
            # pred_grp (B, n_frames, N, G)
            pred_pos, pred_grp = model(
                images[:, j:j + config.n_frames, ...].to(_DEVICE))
            pred_pos = pred_pos.permute(0, 1, 3, 2)

            if j == 0:
                # append all frames in the first output
                for k in range(pred_pos.shape[1]):
                    last_frames_pos.append(pred_pos[
                        :, k, ...].detach().cpu().numpy())   # (B, N, 3)
                    last_frames_grp.append(pred_grp[
                        :, k, ...].detach().cpu().numpy())   # (B, N, G)

            else:
                last_frames_pos.append(pred_pos[
                    :, -1, ...].detach().cpu().numpy())   # (B, N, 3)
                last_frames_grp.append(pred_grp[
                    :, -1, ...].detach().cpu().numpy())   # (B, N, G)

        # last_frames_pos (B, T, N, 3)
        # last_frames_grp (B, T, N, G)
        last_frames_pos = np.stack(last_frames_pos, axis=1)
        last_frames_grp = np.stack(last_frames_grp, axis=1)

        for j in range(n_sections):

            if config.vis_eval and j % config.vis_eval_every == 0:
                # visualize output
                vis_pos = last_frames_pos[
                    0, j:j+config.n_frames_eval, ...]  # (t, N, 3)
                vis_grp = last_frames_grp[
                    0, j:j+config.n_frames_eval, ...]  # (t, N, G)
                visualize_outputs(vis_pos, vis_grp, j, vis_save_dir, zlim)

            for i in range(B):
                # create save directory for all the data in this batch
                data_id = n_traj + i
                data_dir = os.path.join(save_dir, str(data_id))
                if not os.path.isdir(data_dir):
                    os.makedirs(data_dir)

                t_pos = last_frames_pos[
                    i, j:j+config.n_frames_eval, ...]  # (t, N, 3)
                t_grp = last_frames_grp[
                    i, j:j+config.n_frames_eval, ...]  # (t, N, G)

                h5f = h5py.File(os.path.join(data_dir, str(j) + '.h5'), 'w')
                h5f.create_dataset('positions', data=t_pos)
                h5f.create_dataset('groups', data=t_grp)

    return n_traj + B


def main(args):
    config = cfg.get_default()
    cfg.set_params(config, args.config_path, args.set)
    cfg.freeze(config, True)
    print('- Configuration:')
    print(config)

    if config.dataset == 'FluidIceShake':
        n_groups = 2
        n_particles = 348
    elif config.dataset == 'RigidFall':
        n_groups = 3
        n_particles = 192
    elif config.dataset == 'MassRope':
        n_groups = 2
        n_particles = 95
    else:
        raise ValueError('Unsupported environment')

    # generate outputs for both train and valid data
    train_loader = get_dataloader(config, 'train', shuffle=False)
    valid_loader = get_dataloader(config, 'valid', shuffle=False)

    # build model
    model = PointSetNet(
        config.n_frames,
        config.pred_hidden,
        n_particles,
        n_groups,
        config.batchnorm,
        single_out=False,
        recur_pred=config.recur_pred).to(_DEVICE)

    # a model checkpoint must be loaded
    if config.load_path != '':
        print('- Loading model from {}'.format(config.load_path))

        # load model on GPU/CPU
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(config.load_path))
        else:
            model.load_state_dict(
                torch.load(config.load_path, map_location='cpu'))

    else:
        raise ValueError('- Please provide a valid checkpoint')

    if config.log_eval:
        # [[train_data_loss], [valid_data_loss]]
        losses = []

    for loader, name in [(train_loader, 'train'), (valid_loader, 'valid')]:
        # load data with progress bar
        pbar = tqdm(loader)
        n_traj = 0

        # create directory to save output data
        save_dir = os.path.join(config.run_dir, 'eval', name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if config.vis_eval:
            vis_save_dir = os.path.join(save_dir, 'out_vis')
            if not os.path.isdir(vis_save_dir):
                os.makedirs(vis_save_dir)

        if config.log_eval:
            # [(loss, pos_loss, grp_loss) for all data in current loader]
            loader_loss = []

        for images, positions, groups in pbar:
            if config.log_eval:
                model, _, loss, pos_loss, grp_loss = step(
                    config, model, None, images, positions, groups, False)
                loader_loss.append((loss, pos_loss, grp_loss))
                pbar.set_description('Loss {:f}'.format(loss))

            pbar.set_description('Generating video outputs')
            n_traj = generate_outputs(config, model, n_traj, images, save_dir)

            if config.vis:
                visualize(config, model, n_traj // config.batch_size,
                          n_particles, images, positions, groups, False)

        if config.log_eval:
            losses.append(loader_loss)

    if config.log_eval:
        # save all losses into JSON file
        stats = {}
        train_losses, valid_losses = losses
        (stats['train_losses'],
         stats['train_pos_losses'],
         stats['train_grp_losses']) = list(zip(*train_losses))
        (stats['valid_losses'],
         stats['valid_pos_losses'],
         stats['valid_grp_losses']) = list(zip(*valid_losses))

        with open(os.path.join(config.run_dir,
                               'eval_stats.json'), 'w') as fout:
            json.dump(stats, fout)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
