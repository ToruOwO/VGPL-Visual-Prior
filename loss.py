import torch


def chamfer_loss_temporal(x, y):
    """
    Temporal chamfer loss.

    Input:
        x: tensor of shape (B, T, 3, N)
        y: tensor of shape (B, T, 3, N)
    """
    B, T, _, N = x.shape

    x = x.view(B, 3*T, N)
    x = x.permute(0, 2, 1)
    x = torch.unsqueeze(x, 1)
    x = x.repeat(1, N, 1, 1)
    x = x.transpose(1, 2)

    y = y.view(B, 3*T, N)
    y = y.permute(0, 2, 1)
    y = torch.unsqueeze(y, 1)
    y = y.repeat(1, N, 1, 1)

    dist = torch.norm(torch.add(x, -y), dim=-1)
    dist_xy = torch.mean(torch.min(dist, dim=-1)[0])
    dist_yx = torch.mean(torch.min(dist, dim=-2)[0])

    return dist_xy + dist_yx


def chamfer_loss_frame_wise(x, y):
    """
    Frame-wise chamfer loss.

    Input:
        x: tensor of shape (B, T, 3, N)
        y: tensor of shape (B, T, 3, N)
    """
    N = x.shape[-1]

    x = x.permute(0, 1, 3, 2)
    x = torch.unsqueeze(x, 2)
    x = x.repeat(1, 1, N, 1, 1)
    x = x.transpose(2, 3)

    y = y.permute(0, 1, 3, 2)
    y = torch.unsqueeze(y, 2)
    y = y.repeat(1, 1, N, 1, 1)

    dist = torch.norm(torch.add(x, -y), dim=-1)
    dist_xy = torch.mean(torch.min(dist, dim=-1)[0])
    dist_yx = torch.mean(torch.min(dist, dim=-2)[0])

    return dist_xy + dist_yx


def chamfer_loss_with_grouping(x, gx, y, gy):
    """
    Temporal chamfer loss with grouping
    Label must be turned into one-hot vectors before hand

    Input:
        x: tensor of shape (B, T, 3, N)
        gx: tensor of shape (B, T, N, G)
        y: tensor of shape (B, T, 3, N)
        gy: tensor of shape (B, T, N, G)
    """
    B, T, _, N = x.shape

    x = x.view(B, 3*T, N)
    x = x.permute(0, 2, 1)
    x = torch.unsqueeze(x, 1)
    x = x.repeat(1, N, 1, 1)
    x = x.transpose(1, 2)

    y = y.view(B, 3*T, N)
    y = y.permute(0, 2, 1)
    y = torch.unsqueeze(y, 1)
    y = y.repeat(1, N, 1, 1)

    gx = gx.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B, N, T*G)
    gx = torch.unsqueeze(gx, 1)
    gx = gx.repeat(1, N, 1, 1)
    gx = gx.transpose(1, 2)

    gy = gy.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B, N, T*G)
    gy = torch.unsqueeze(gy, 1)
    gy = gy.repeat(1, N, 1, 1)

    dist = torch.norm(torch.add(x, -y), dim=-1)**2
    min_dist_xy, min_dist_xy_idxs = torch.min(dist, dim=-1)
    min_dist_yx, min_dist_yx_idxs = torch.min(dist, dim=-2)

    grp_dist = torch.norm(torch.add(gx, -gy), dim=-1)**2
    grp_dist_xy = torch.cat(
        [grp_dist[i].index_select(
            -1, min_dist_xy_idxs[i]).diagonal().unsqueeze(0)
         for i in range(B)], 0)
    grp_dist_yx = torch.cat(
        [grp_dist[i].index_select(
            -2, min_dist_yx_idxs[i]).diagonal().unsqueeze(0)
         for i in range(B)], 0)

    pos_loss = min_dist_xy.mean() + min_dist_yx.mean()
    grp_loss = grp_dist_xy.mean() + grp_dist_yx.mean()

    return pos_loss, grp_loss
