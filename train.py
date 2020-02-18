"""Script for model training."""

import argparse
import json
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
import utils
from dataset import denormalize, get_dataloader
from loss import chamfer_loss_with_grouping
from model import PointSetNet, get_pos_diff

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--set', nargs='+')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch_data(input_array, idxs, i, n_frames):
    """
    Get batched array from random shuffled indices (idxs).

    Input:
        input_array: (B, T, ...)
        idxs: (B, T - n_frames, ...)
        i: int
        n_frames: int
    """
    step_array = None
    for b in range(input_array.shape[0]):
        if step_array is None:
            step_array = input_array[
                b:b + 1, idxs[b][i]:(idxs[b][i] + n_frames), ...]
        else:
            step_array = torch.cat(
                (step_array,
                 input_array[b:b + 1,
                             idxs[b][i]:(idxs[b][i] + n_frames), ...]))
    return step_array


def step(config, model, optimizer, images, positions, groups, is_train):
    """
    Model takes a forward step.

    Input:
        images: (B, T, C, H, W)
        positions: (B, T, N, 3)
        groups: (B, N)
    """
    _, _, C, H, W = images.shape
    B, T, N, _ = positions.shape

    if is_train:
        model.train()
    else:
        model.eval()

    # stack groups (B, N) -> (B, T, N)
    groups = groups.unsqueeze(1).repeat(1, config.n_frames, 1)

    # shuffle on the time dimension
    if is_train:
        step_idxs = np.stack(
            [np.random.permutation(T - config.n_frames) for b in range(B)])
    else:
        step_idxs = np.stack(
            [np.arange(T - config.n_frames) for b in range(B)])

    batch_loss = 0
    batch_group_loss = 0
    batch_position_loss = 0
    for i in range(step_idxs.shape[1]):
        # Send to GPU after slicing to save memory
        step_images = get_batch_data(
            images, step_idxs, i, config.n_frames).to(_DEVICE)
        step_positions = get_batch_data(
            positions, step_idxs, i,
            config.n_frames).transpose(2, 3).to(_DEVICE)
        step_groups = groups.long().to(_DEVICE)

        # pred_pos (B, 1, 3, N) if single_out else (B, T, 3, N)
        # pred_grp (B, 1, N, G) if single_out else (B, T, N, G)
        pred_pos, pred_grp = model(step_images)

        # only use last frame from ground truth to calculate loss
        if config.single_out \
           or (not config.recur_pred and config.use_temp_encoder):
            # (B, T, 3, N) -> (B, 1, 3, N)
            gt_positions = step_positions[:, -1, ...].unsqueeze(1)
            # (B, T, N) -> (B, 1, N)
            gt_groups = step_groups[:, -1, ...].unsqueeze(1)
        else:
            gt_positions = step_positions
            gt_groups = step_groups

        if config.loss_type == 'l2':
            pos_loss = F.mse_loss(gt_positions, pred_pos)

            # pred_grp (B, T, N, G) -> (N, G, B, T)
            pred_grp = pred_grp.permute(2, 3, 0, 1)
            # gt_groups (B, T, N) -> (N, B, T)
            gt_groups = gt_groups.permute(2, 0, 1)
            grp_loss = F.cross_entropy(pred_grp, gt_groups)
        elif config.loss_type == 'chamfer':
            gt_groups = torch.zeros_like(pred_grp).scatter_(
                -1, gt_groups.unsqueeze(-1), 1)  # (B, T, N, G)
            pos_loss, grp_loss = chamfer_loss_with_grouping(
                gt_positions, gt_groups, pred_pos, pred_grp)
        else:
            raise ValueError(
                'Invalid loss type {}'.format(config.loss_type))

        # trajectory smoothness regularizer
        pos_loss += config.temp_reg_lam * get_pos_diff(pred_pos).norm()

        loss = pos_loss + grp_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_loss += loss.item()
        batch_group_loss += grp_loss.item()
        batch_position_loss += pos_loss.item()

    batch_loss /= step_idxs.shape[1]
    batch_group_loss /= step_idxs.shape[1]
    batch_position_loss /= step_idxs.shape[1]

    return model, optimizer, batch_loss, batch_position_loss, batch_group_loss


def visualize(config, model, epoch, n_particles, images,
              positions, groups, is_train, is_eval=False):
    """
    images: (B, T, C, H, W)
    positions: (B, T, N, 3)
    groups: (B, N)
    """
    dir_split = 'train' if is_train else 'valid'
    if is_eval:
        save_dir = os.path.join(
            config.run_dir, 'eval', 'batch_{}'.format(epoch))
    else:
        save_dir = os.path.join(
            config.run_dir, 'vis', 'epoch{:03d}_{}'.format(epoch, dir_split))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model.eval()

    _, _, C, H, W = images.shape
    B = config.vis_samples
    T_vis = config.vis_frames_per_sample

    vis_gt_pos, vis_pred_pos, vis_pred_grp = None, None, None
    for j in range(T_vis):
        vis_images = images[:B, j:j + config.n_frames,
                            ...].to(_DEVICE)
        gt_pos = positions[:B, j:j + config.n_frames,
                           ...].transpose(2, 3).to(_DEVICE)
        pred_pos, pred_grp = model(vis_images)

        # Take the last time step
        new_gt_pos = gt_pos[:, -1, ...].detach().cpu().unsqueeze(1).numpy()
        new_pred_pos = pred_pos[:, -1, ...].detach().cpu().unsqueeze(1).numpy()
        new_pred_grp = pred_grp[:, -1, ...].detach().cpu().unsqueeze(1).numpy()
        if vis_gt_pos is None:
            vis_gt_pos = new_gt_pos
            vis_pred_pos = new_pred_pos
            vis_pred_grp = new_pred_grp
        else:
            vis_gt_pos = np.concatenate(
                (vis_gt_pos, new_gt_pos), 1)
            vis_pred_pos = np.concatenate(
                (vis_pred_pos, new_pred_pos), 1)
            vis_pred_grp = np.concatenate(
                (vis_pred_grp, new_pred_grp), 1)

    for k in range(B):
        vis_input = denormalize(
            images[k, config.n_frames - 1:config.n_frames + T_vis - 1,
                   ...].cpu().numpy())
        vis_input = (vis_input * 255).astype('uint8')

        # randomly select 5 particles to highlight with
        # different colors for tracking temporal consistency
        highlight_idxs = np.random.choice(
            range(n_particles), 5, replace=False)

        # use special z-axis range for MassRope visualization
        if config.dataset == 'MassRope':
            zlim = (0.3, 1.7)
        else:
            zlim = (0, 0.7)

        # visualize ground truth groups (N,)
        vis_gt_frames = utils.toy_render(
            np.transpose(vis_gt_pos[k], (0, 2, 1)),
            title='Ground Truth',
            gt_groups=groups[k],
            zlim=zlim)

        vis_pred_frames = utils.toy_render(
            np.transpose(vis_pred_pos[k], (0, 2, 1)),
            title='Prediction',
            highlight_idxs=highlight_idxs,
            groups=vis_pred_grp[k],
            zlim=zlim)

        vis_frames = np.concatenate(
            (vis_gt_frames, vis_pred_frames), 2)

        imageio.mimwrite(
            os.path.join(
                save_dir, 'sample{:02d}_input.mp4'.format(k)),
            vis_input,
            fps=15)
        imageio.mimwrite(
            os.path.join(
                save_dir,
                'sample{:02d}_particles.mp4'.format(k)),
            vis_frames,
            fps=15)


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

    train_loader = get_dataloader(config, 'train')
    valid_loader = get_dataloader(config, 'valid')

    # build model
    model = PointSetNet(
        config.n_frames,
        config.pred_hidden,
        n_particles,
        n_groups,
        config.batchnorm,
        single_out=config.single_out,
        recur_pred=config.recur_pred,
        use_temp_encoder=config.use_temp_encoder,
        conv_temp_encoder=config.conv_temp_encoder,
        temp_embedding_size=config.temp_embedding_size).to(_DEVICE)

    print('- Model architecture:')
    print(model)

    if config.load_path != '':
        print('- Loading model from {}'.format(config.load_path))
        model.load_state_dict(torch.load(config.load_path))

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if not config.debug:
        print('- Training start')
        stats = {
            'epoch': [],
            'valid_losses': [],
            'train_losses': [],
            'train_pos_losses': [],
            'train_group_losses': [],
            'valid_pos_losses': [],
            'valid_group_losses': [],
        }
        best_valid_loss = np.Inf

        for epoch in range(config.n_epochs):

            # training
            print('- Training epoch {:d}'.format(epoch))
            epoch_train_losses = []
            epoch_train_pos_losses = []
            epoch_train_grp_losses = []

            pbar = tqdm(train_loader)
            did_vis = False
            for images, positions, groups in pbar:
                (model, optimizer, train_loss,
                 train_pos_loss, train_grp_loss) = step(
                    config, model, optimizer, images, positions, groups, True)
                epoch_train_losses.append(train_loss)
                epoch_train_pos_losses.append(train_pos_loss)
                epoch_train_grp_losses.append(train_grp_loss)
                pbar.set_description('Train loss {:f}'.format(train_loss))

                # visualize training results
                if not did_vis \
                        and config.vis and (epoch + 1) % config.vis_every == 0:
                    pbar.set_description('Generating video')
                    visualize(config, model, epoch, n_particles,
                              images, positions, groups, True)
                    did_vis = True

            train_loss = np.average(epoch_train_losses)
            train_pos_loss = np.average(epoch_train_pos_losses)
            train_grp_loss = np.average(epoch_train_grp_losses)

            print(('- Finish training epoch {:d}, training loss {:f},'
                   ' pos loss {:f}, group loss {:f}').format(
                epoch, train_loss, train_pos_loss, train_grp_loss))

            # validation
            print('- Evaluating epoch {:d}'.format(epoch))
            epoch_valid_losses = []
            epoch_valid_pos_losses = []
            epoch_valid_grp_losses = []

            pbar = tqdm(valid_loader)
            did_vis = False
            for images, positions, groups in pbar:
                with torch.no_grad():
                    (model, _, valid_loss,
                     valid_pos_loss, valid_grp_loss) = step(
                        config, model, optimizer,
                        images, positions, groups, False)
                epoch_valid_losses.append(valid_loss)
                epoch_valid_pos_losses.append(valid_pos_loss)
                epoch_valid_grp_losses.append(valid_grp_loss)
                pbar.set_description('Valid loss {:f}'.format(valid_loss))

                # visualize validation results
                if not did_vis \
                        and config.vis and (epoch + 1) % config.vis_every == 0:
                    pbar.set_description('Generating video')
                    visualize(config, model, epoch, n_particles,
                              images, positions, groups, False)
                    did_vis = True

            valid_loss = np.average(epoch_valid_losses)
            valid_pos_loss = np.average(epoch_valid_pos_losses)
            valid_grp_loss = np.average(epoch_valid_grp_losses)

            print('- Finish eval epoch {:d}, validation loss {:f}'.format(
                epoch, valid_loss))
            if valid_loss < best_valid_loss:
                print('- Best model')
                best_valid_loss = valid_loss
                torch.save(model.state_dict(),
                           os.path.join(config.run_dir, 'checkpoint_best.pth'))
            torch.save(model.state_dict(),
                       os.path.join(config.run_dir, 'checkpoint_latest.pth'))
            print()

            stats['epoch'].append(epoch)
            stats['train_losses'].append(train_loss)
            stats['valid_losses'].append(valid_loss)
            stats['train_pos_losses'].append(train_pos_loss)
            stats['train_group_losses'].append(train_grp_loss)
            stats['valid_pos_losses'].append(valid_pos_loss)
            stats['valid_group_losses'].append(valid_grp_loss)
            with open(os.path.join(config.run_dir, 'stats.json'), 'w') as fout:
                json.dump(stats, fout)

            # Plot loss curves
            plot_dir = os.path.join(config.run_dir, 'curves')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            utils.plot_curves(
                x=stats['epoch'],
                ys=[stats['train_losses'], stats['valid_losses']],
                save_path=os.path.join(plot_dir, 'loss.png'),
                curve_labels=['train', 'valid'],
                x_label='epoch',
                y_label='total_loss',
                title='Total loss')
            utils.plot_curves(
                x=stats['epoch'],
                ys=[stats['train_pos_losses'], stats['valid_pos_losses']],
                save_path=os.path.join(plot_dir, 'loss_pos.png'),
                curve_labels=['train', 'valid'],
                x_label='epoch',
                y_label='pos_loss',
                title='Position loss')
            utils.plot_curves(
                x=stats['epoch'],
                ys=[stats['train_group_losses'], stats['valid_group_losses']],
                save_path=os.path.join(plot_dir, 'loss_grp.png'),
                curve_labels=['train', 'valid'],
                x_label='epoch',
                y_label='grp_loss',
                title='Grouping loss')

    else:  # Debug on a single batch
        images, positions, groups = next(iter(train_loader))
        images = images[:5, :15, ...]
        positions = positions[:5, :15, ...]
        groups = groups[:5, ...]
        for epoch in range(config.n_epochs):
            (model, optimizer, train_loss,
             train_pos_loss, train_grp_loss) = step(
                config, model, optimizer, images, positions, groups, True)
            print(train_loss, train_pos_loss, train_grp_loss)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
