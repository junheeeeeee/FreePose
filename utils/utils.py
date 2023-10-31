import torch
from pytorch3d.transforms import so3_exponential_map as rodrigues
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def p3d_no_scale(p3d):
    p3d = p3d.reshape(-1, 3, 16)
    hey = p3d * 1

    bone_inx = [6, 0, 1, 6, 3, 4, -1, 8, 9, -1, 7, 10, 11, 7, 13, 14]

    bone_lenth = torch.zeros((p3d.shape[0], 14)).cuda()
    n = 0
    for i, j in enumerate(bone_inx):
        if j == -1:
            pass
        else:
            bone_lenth[:, n] = ((hey[:, :, j] - hey[:, :, i]) ** 2).sum(-1) ** 0.5
            n += 1
    scale_p3d = bone_lenth.mean(-1).unsqueeze(1).unsqueeze(1)

    p3d_scaled = p3d / scale_p3d

    # loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return p3d_scaled

def rand_position(poses, rott=None, trans=None):
    poses = poses.detach().reshape(-1, 3, 16)

    poses = poses - poses[:, :, 6][:, :, None]

    if rott == None:

        rand_size = torch.rand((poses.shape[0], 1)).cuda() * torch.pi * 2
        rand_vec = (torch.ones((poses.shape[0], 3))).cuda()
        rand_theta = torch.rand((poses.shape[0])).cuda() * torch.pi
        rand_pie = torch.rand((poses.shape[0])).cuda() * torch.pi * 2
        rand_vec[:, 0] *= torch.sin(rand_theta) * torch.cos(rand_pie)
        rand_vec[:, 1] *= torch.sin(rand_theta) * torch.sin(rand_pie)
        rand_vec[:, 2] *= torch.cos(rand_theta)
        rand_vec = rand_vec * rand_size
        rott = rodrigues(rand_vec)


    poses = (rott @ poses)

    if trans == None:
        rand_num = torch.rand((poses.shape[0], 3)).cuda()
        trans = rand_num[:, :, None] * 1

        trans[:, 0] = torch.randn(trans[:, 0].shape) * 1500
        trans[:, 1] = torch.randn(trans[:, 0].shape) * 1500
        trans[:, 2] = trans[:, 2] * 5000 + 1500
    poses += trans
    f = torch.rand((poses.shape[0], 1, 1)).cuda() * 1500 + 1000
    return poses, (poses[:, :2] / poses[:, 2][:, None]* f / 512).reshape(-1, 32), rott, trans

def human_model(pred):
    lenth_rate = 10

    pred = pred.unsqueeze(-1)

    len = np.array([0.49108774, 1.80588788, 0.43735805, 0.4342996 , 0.55776042,
       1.03019458, 0.92972383, 1.63597411, 1.6777138 ]) * 270
    p = torch.zeros((pred.shape[0], 17, 3)).cuda()
    x = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([1, 0, 0]).cuda()
    y = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([0, 1, 0]).cuda()
    z = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([0, 0, 1]).cuda()

    p[:, 1] = p[:, 1] - x * len[0] * (1 + pred[:, 0] / lenth_rate)
    p[:, 4] = p[:, 4] + x * len[0] * (1 + pred[:, 0] / lenth_rate)

    p[:, 8] = p[:, 8] + z * len[1] * (1 + pred[:, 1] / lenth_rate)

    head_dot = len[2] * (1 + pred[:, 32] / lenth_rate)
    head_top = len[3] * (1 + pred[:, 33] / lenth_rate)
    nose = (head_dot / 2 + head_top / 2) * 7 / 12 * (0.8 + abs(pred[:, 2]) / 5)
    p[:, 9] = p[:, 8] + y * nose + z * (head_dot ** 2 - nose ** 2) ** 0.5
    p[:, 10] = p[:, 9] - y * nose + z * (head_top ** 2 - nose ** 2) ** 0.5

    head_rot = rodrigues(z * ((pred[:, 8]) * 45) * torch.pi / 180)
    head_front = rodrigues(-x * (pred[:, 9] * 60 + 10) * torch.pi / 180)
    head_side = rodrigues(y * ((pred[:, 10]) * 35) * torch.pi / 180)
    p[:, 9:11] = torch.transpose(
        head_front @ head_side @ head_rot @ torch.transpose(p[:, 9:11] - p[:, 8][:, None, :], 1, 2), 1, 2) + p[:, 8][:,
                                                                                                             None, :]

    shod_l_rot = rodrigues(y * torch.arcsin((pred[:, 11] + 1) * 0.8 / 2))
    shod_len = (x * len[4] * (1 + pred[:, 3] / 3))[:, :, None]
    shod_l = shod_l_rot.matmul(shod_len).squeeze(-1)
    p[:, 11] = p[:, 8] + shod_l
    shod_r_rot = rodrigues(-y * torch.arcsin((pred[:, 12] + 1) * 0.8 / 2))
    shod_r = shod_r_rot.matmul(shod_len).squeeze(-1)
    p[:, 14] = p[:, 8] - shod_r

    p[:, 12] = p[:, 11] - z * len[5] * (1 + pred[:, 4] / lenth_rate)
    p[:, 15] = p[:, 14] - z * len[5] * (1 + pred[:, 4] / lenth_rate)

    elbow_l_rot = rodrigues(x * (pred[:, 13] + 1) / 2 * 3.14 * 145 / 180)

    elbow_l_rot2 = rodrigues(z * ((pred[:, 14] + 1) / 2 * 150 - 40) * 3.14 / 180)

    elbow_l = (elbow_l_rot2 @ elbow_l_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 13] = p[:, 12] - elbow_l * 251.7 * (1 + pred[:, 5] / 3)

    # side_angel = ((pred[:, 15] + 1) / 2 * 140 - 90)
    side_angel = ((pred[:, 15] + 1) / 2 * 170 - 120)
    front_angel = ((pred[:, 16] + 1) / 2 * 225 - 45)
    # side_angel = torch.where(front_angel < 0, (side_angel + 130) * 9 / 19, side_angel)
    # side_angel = torch.where(front_angel >= 165, side_angel * 0, side_angel)
    # side_angel = torch.where(abs(front_angel) <= 15, side_angel * 0, side_angel)

    shod_l_front = rodrigues(x * front_angel * 3.14 / 180)

    shod_vec = (shod_l_front @ y[:, :, None])[:, :, 0]

    shod_l_side = rodrigues(shod_vec * side_angel * 3.14 / 180)

    p[:, 12:14] = torch.transpose(
        shod_l_side @ shod_l_front @ torch.transpose(p[:, 12:14] - p[:, 11][:, None, :], 1, 2), 1, 2) + p[:, 11][:,
                                                                                                        None, :]

    elbow_r_rot = rodrigues(x * (pred[:, 17] + 1) / 2 * 3.14 * 145 / 180)

    elbow_r_rot2 = rodrigues(-z * ((pred[:, 18] + 1) / 2 * 150 - 40) * 3.14 / 180)

    elbow_r = (elbow_r_rot2 @ elbow_r_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 16] = p[:, 15] - elbow_r * len[6] * (1 + pred[:, 5] / lenth_rate)

    # side_angel = ((pred[:, 19] + 1) / 2 * 140 - 90)
    side_angel = ((pred[:, 19] + 1) / 2 * 170 - 120)
    front_angel = ((pred[:, 20] + 1) / 2 * 225 - 45)
    # side_angel = torch.where(front_angel < 0, (side_angel + 130) * 9 / 19, side_angel)
    # side_angel = torch.where(front_angel >= 165, side_angel * 0, side_angel)
    # side_angel = torch.where(abs(front_angel) <= 15, side_angel * 0, side_angel)

    shod_r_front = rodrigues(x * front_angel * 3.14 / 180)

    shod_vec = (shod_r_front @ -y[:, :, None])[:, :, 0]

    shod_r_side = rodrigues(shod_vec * side_angel * 3.14 / 180)

    p[:, 15:17] = torch.transpose(
        shod_r_side @ shod_r_front @ torch.transpose(p[:, 15:17] - p[:, 14][:, None, :], 1, 2), 1, 2) + p[:, 14][:,
                                                                                                        None, :]
    p[:, 2] = p[:, 1] - z * len[7] * (1 + pred[:, 6] / lenth_rate)
    p[:, 5] = p[:, 4] - z * len[7] * (1 + pred[:, 6] / lenth_rate)
    knee_r_rot = rodrigues(-x * (pred[:, 21] + 1) / 2 * torch.pi * 135 / 180)
    knee_r_rot2 = rodrigues(-z * (pred[:, 22] * 45) * torch.pi / 180)
    knee_r = knee_r_rot2.matmul(knee_r_rot.matmul(z[:, :, None])).squeeze(-1)
    p[:, 3] = p[:, 2] - knee_r * len[8] * (1 + pred[:, 7] / lenth_rate)
    knee_r_side = rodrigues(y * ((pred[:, 23] + 1) / 2 * 70 - 25) * torch.pi / 180)

    knee_r_front = rodrigues(x * ((pred[:, 24] + 1) / 2 * 140 - 30) * torch.pi / 180)
    p[:, 1:4] = torch.transpose(knee_r_front @ knee_r_side @ torch.transpose(p[:, 1:4] - p[:, 1][:, None, :], 1, 2), 1,
                                2) + p[:, 1][:, None, :]

    knee_l_rot = rodrigues(-x * (pred[:, 25] + 1) / 2 * torch.pi * 135 / 180)
    knee_l_rot2 = rodrigues(z * (pred[:, 26] * 45) * torch.pi / 180)
    knee_l = knee_l_rot2.matmul(knee_l_rot.matmul(z[:, :, None])).squeeze(-1)
    p[:, 6] = p[:, 5] - knee_l * len[8] * (1 + pred[:, 7] / lenth_rate)
    knee_l_side = rodrigues(-y * ((pred[:, 27] + 1) / 2 * 70 - 25) * torch.pi / 180)
    # knee_l_front = rodrigues(x * ((pred[:, 28] + 1) / 2 * 135 - 45) * torch.pi / 180)
    knee_l_front = rodrigues(x * ((pred[:, 28] + 1) / 2 * 140 - 30) * torch.pi / 180)
    p[:, 5:7] = torch.transpose(knee_l_front @ knee_l_side @ torch.transpose(p[:, 5:7] - p[:, 4][:, None, :], 1, 2), 1,
                                2) + p[:, 4][:, None, :]

    spine_rot = rodrigues(z * ((pred[:, 29]) * 30) * torch.pi / 180)
    spine_front = rodrigues(-x * ((pred[:, 30] + 1) / 2 * 105 - 30) * torch.pi / 180)
    spine_side = rodrigues(y * ((pred[:, 31]) * 35) * torch.pi / 180)
    p[:, 7:] = torch.transpose(spine_front @ spine_side @ spine_rot @ torch.transpose(p[:, 7:] * 1, 1, 2), 1, 2)
    # p[:, :7] = torch.transpose(spine_front @ torch.transpose(p[:, :7] * 1, 1, 2), 1, 2)
    index = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    pp = torch.zeros((pred.shape[0], 16, 3)).cuda()
    for i, j in enumerate(index):
        pp[:, i] = p[:, j]
    pp = torch.transpose(pp, 1, 2).reshape(-1, 3, 16)
    p3d = torch.zeros_like(pp)
    p3d[:, 0] = pp[:, 0]
    p3d[:, 1] = -pp[:, 2]
    p3d[:, 2] = -pp[:, 1]
    return p3d
def human_model3(pred):
    lenth_rate = 10

    pred = pred.unsqueeze(-1)

    len = np.array([0.49108774, 1.80588788, 0.43735805, 0.4342996 , 0.55776042,
       1.03019458, 0.92972383, 1.63597411, 1.6777138 ]) * 270
    p = torch.zeros((pred.shape[0], 17, 3)).cuda()
    x = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([1, 0, 0]).cuda()
    y = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([0, 1, 0]).cuda()
    z = torch.zeros((pred.shape[0], 3)).cuda() + torch.tensor([0, 0, 1]).cuda()

    p[:, 1] = p[:, 1] - x * len[0] * (1 + pred[:, 0] / lenth_rate)
    p[:, 4] = p[:, 4] + x * len[0] * (1 + pred[:, 0] / lenth_rate)

    p[:, 8] = p[:, 8] + z * len[1] * (1 + pred[:, 1] / lenth_rate)

    head_dot = len[2] * (1 + pred[:, 32] / lenth_rate)
    head_top = len[3] * (1 + pred[:, 33] / lenth_rate)
    nose = (head_dot / 2 + head_top / 2) * 7 / 12 * (0.8 + abs(pred[:, 2]) / 5)
    p[:, 9] = p[:, 8] + y * nose + z * (head_dot ** 2 - nose ** 2) ** 0.5
    p[:, 10] = p[:, 9] - y * nose + z * (head_top ** 2 - nose ** 2) ** 0.5

    head_rot = rodrigues(z * ((pred[:, 8]) * 45) * torch.pi / 180)
    head_front = rodrigues(-x * (pred[:, 9] * 60 + 10) * torch.pi / 180)
    head_side = rodrigues(y * ((pred[:, 10]) * 35) * torch.pi / 180)
    p[:, 9:11] = torch.transpose(
        head_front @ head_side @ head_rot @ torch.transpose(p[:, 9:11] - p[:, 8][:, None, :], 1, 2), 1, 2) + p[:, 8][:,
                                                                                                             None, :]

    shod_l_rot = rodrigues(y * torch.arcsin((pred[:, 11] + 1) * 0.8 / 2))
    shod_len = (x * len[4] * (1 + pred[:, 3] / 3))[:, :, None]
    shod_l = shod_l_rot.matmul(shod_len).squeeze(-1)
    p[:, 11] = p[:, 8] + shod_l
    shod_r_rot = rodrigues(-y * torch.arcsin((pred[:, 12] + 1) * 0.8 / 2))
    shod_r = shod_r_rot.matmul(shod_len).squeeze(-1)
    p[:, 14] = p[:, 8] - shod_r

    p[:, 12] = p[:, 11] - z * len[5] * (1 + pred[:, 4] / lenth_rate)
    p[:, 15] = p[:, 14] - z * len[5] * (1 + pred[:, 4] / lenth_rate)

    elbow_l_rot = rodrigues(x * (pred[:, 13] + 1) / 2 * 3.14 * 145 / 180)

    elbow_l_rot2 = rodrigues(z * ((pred[:, 14] + 1) / 2 * 150 - 40) * 3.14 / 180)

    elbow_l = (elbow_l_rot2 @ elbow_l_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 13] = p[:, 12] - elbow_l * 251.7 * (1 + pred[:, 5] / 3)

    # side_angel = ((pred[:, 15] + 1) / 2 * 140 - 90)
    side_angel = ((pred[:, 15] + 1)/ 2 * 190 - 170)
    front_angel = ((pred[:, 16] + 1) / 2 * 225 - 45)
    # side_angel = torch.where(abs(front_angel) < 30, -abs(side_angel), side_angel)
    # side_angel = torch.where(front_angel < 0, (side_angel + 130) * 9 / 19, side_angel)
    # side_angel = torch.where(front_angel >= 165, side_angel * 0, side_angel)
    # side_angel = torch.where(abs(front_angel) <= 15, side_angel * 0, side_angel)

    shod_l_side = rodrigues(y * side_angel * 3.14 / 180)


    shod_vec = (shod_l_side @ x[:, :, None])[:, :, 0]

    shod_l_front = rodrigues(shod_vec * front_angel * 3.14 / 180)

    p[:, 12:14] = torch.transpose(
        shod_l_front @ shod_l_side @ torch.transpose(p[:, 12:14] - p[:, 11][:, None, :], 1, 2), 1, 2) + p[:, 11][:,
                                                                                                        None, :]


    elbow_r_rot = rodrigues(x * (pred[:, 17] + 1) / 2 * 3.14 * 145 / 180)

    elbow_r_rot2 = rodrigues(-z * ((pred[:, 18] + 1) / 2 * 150 - 40) * 3.14 / 180)

    elbow_r = (elbow_r_rot2 @ elbow_r_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 16] = p[:, 15] - elbow_r * len[6] * (1 + pred[:, 5] / lenth_rate)

    # side_angel = ((pred[:, 19] + 1) / 2 * 140 - 90)
    side_angel = ((pred[:, 19] + 1) / 2 * 190 - 170)
    # side_angel = nonlinear(pred[:, 19], 20, -170)

    front_angel = ((pred[:, 20] + 1) / 2 * 225 - 45)
    # front_angel = nonlinear(pred[:, 20], 180, -45 )
    # side_angel = torch.where(front_angel < 40, -abs(side_angel), side_angel)
    # front_angel = torch.where(side_angel < 0, torch.where(abs(front_angel)<40, front_angel + 20 * front_angel/abs(front_angel), front_angel), front_angel)
    # exit()
    # side_angel = torch.where(front_angel < 0, (side_angel + 130) * 9 / 19, side_angel)
    # side_angel = torch.where(front_angel >= 165, side_angel * 0, side_angel)
    # side_angel = torch.where(abs(front_angel) <= 15, side_angel * 0, side_angel)

    # shod_r_front = rodrigues(x * front_angel * 3.14 / 180)
    #
    # shod_vec = (shod_r_front @ -y[:, :, None])[:, :, 0]
    #
    # shod_r_side = rodrigues(shod_vec* side_angel * 3.14 / 180)
    # p[:, 15:17] = torch.transpose(
    #   shod_r_side @ shod_r_front @ torch.transpose(p[:, 15:17] - p[:, 14][:, None, :], 1, 2), 1, 2) + p[:, 14][:,
    #                                                                                                     None, :]
    shod_r_side = rodrigues(-y * side_angel * 3.14 / 180)
    shod_vec = (shod_r_side @ x[:, :, None])[:, :, 0]
    shod_r_front = rodrigues(shod_vec * front_angel * 3.14 / 180)
    shod_r = rodrigues( -y * side_angel * 3.14 / 180 + x * front_angel * 3.14 / 180)

    # p[:, 15:17] = torch.transpose(
    #     shod_r @ torch.transpose(p[:, 15:17] - p[:, 14][:, None, :], 1, 2), 1, 2) + p[:, 14][:,
    #                                                                                                     None, :]
    p[:, 15:17] = torch.transpose(
        shod_r_front @ shod_r_side @ torch.transpose(p[:, 15:17] - p[:, 14][:, None, :], 1, 2), 1, 2) + p[:, 14][:,
                                                                                                        None, :]
    p[:, 2] = p[:, 1] - z * len[7] * (1 + pred[:, 6] / lenth_rate)
    p[:, 5] = p[:, 4] - z * len[7] * (1 + pred[:, 6] / lenth_rate)
    knee_r_rot = rodrigues(-x * (pred[:, 21] + 1) / 2 * torch.pi * 135 / 180)
    knee_r_rot2 = rodrigues(-z * (pred[:, 22] * 45) * torch.pi / 180)
    knee_r = knee_r_rot2.matmul(knee_r_rot.matmul(z[:, :, None])).squeeze(-1)
    p[:, 3] = p[:, 2] - knee_r * len[8] * (1 + pred[:, 7] / lenth_rate)
    knee_r_side = rodrigues(y * ((pred[:, 23] + 1) / 2 * 70 - 25) * torch.pi / 180)

    knee_r_front = rodrigues(x * ((pred[:, 24] + 1) / 2 * 140 - 30) * torch.pi / 180)
    p[:, 1:4] = torch.transpose(knee_r_front @ knee_r_side @ torch.transpose(p[:, 1:4] - p[:, 1][:, None, :], 1, 2), 1,
                                2) + p[:, 1][:, None, :]

    knee_l_rot = rodrigues(-x * (pred[:, 25] + 1) / 2 * torch.pi * 135 / 180)
    knee_l_rot2 = rodrigues(z * (pred[:, 26] * 45) * torch.pi / 180)
    knee_l = knee_l_rot2.matmul(knee_l_rot.matmul(z[:, :, None])).squeeze(-1)
    p[:, 6] = p[:, 5] - knee_l * len[8] * (1 + pred[:, 7] / lenth_rate)
    knee_l_side = rodrigues(-y * ((pred[:, 27] + 1) / 2 * 70 - 25) * torch.pi / 180)
    # knee_l_front = rodrigues(x * ((pred[:, 28] + 1) / 2 * 135 - 45) * torch.pi / 180)
    knee_l_front = rodrigues(x * ((pred[:, 28] + 1) / 2 * 140 - 30) * torch.pi / 180)
    p[:, 5:7] = torch.transpose(knee_l_front @ knee_l_side @ torch.transpose(p[:, 5:7] - p[:, 4][:, None, :], 1, 2), 1,
                                2) + p[:, 4][:, None, :]

    spine_rot = rodrigues(z * ((pred[:, 29]) * 30) * torch.pi / 180)
    spine_front = rodrigues(-x * ((pred[:, 30] + 1) / 2 * 105 - 30) * torch.pi / 180)
    spine_side = rodrigues(y * ((pred[:, 31]) * 35) * torch.pi / 180)
    p[:, 7:] = torch.transpose(spine_front @ spine_side @ spine_rot @ torch.transpose(p[:, 7:] * 1, 1, 2), 1, 2)
    # p[:, :7] = torch.transpose(spine_front @ torch.transpose(p[:, :7] * 1, 1, 2), 1, 2)
    index = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    pp = torch.zeros((pred.shape[0], 16, 3)).cuda()
    for i, j in enumerate(index):
        pp[:, i] = p[:, j]
    pp = torch.transpose(pp, 1, 2).reshape(-1, 3, 16)
    p3d = torch.zeros_like(pp)
    p3d[:, 0] = pp[:, 0]
    p3d[:, 1] = -pp[:, 2]
    p3d[:, 2] = -pp[:, 1]
    return p3d
def get_distance(pose):

    torso = torch.linalg.norm((pose[:,:,6] - pose[:,:,3]), dim=1) * 0.9
    up_arm =  torch.linalg.norm((pose[:,:,13] - pose[:,:,14]), dim=1) / 16
    down_arm = torch.linalg.norm((pose[:,:,14] - pose[:,:,15]), dim=1) / 16
    up_leg = torch.linalg.norm((pose[:, :, 0] - pose[:, :, 1]), dim=1) / 16
    down_leg = torch.linalg.norm((pose[:, :, 1] - pose[:, :, 2]), dim=1) / 16
    a = torch.cross((pose[:, :, 7] - pose[:, :, 8]),(pose[:, :, 7] - pose[:, :, 9]))
    head = torch.linalg.norm(a, dim=1) / torch.linalg.norm((pose[:, :, 7] - pose[:, :, 9]), dim=1) * 2
    shoulder = up_arm * 1
    pelvis = up_leg * 1
    # limit = torch.tensor(
    #     [25, 25, 25, 25, 25, 25, 25, 25, 100, 25, 5, 5, 15, 10, 15, 10, 100, 25, 100, 100, 25, 5, 0, 15, 10, 15, 10, 100,
    #      25, 100, 100, 25, 25, 25, 25, 25, 25, 100, 15, 10, 15, 10, 100, 25, 100, 100, 15, 10, 15, 10, 100, 25, 100, 100, 15,
    #      15, 100, 100, 100, 15, 0, 100, 25, 100, 100, 100, 100, 100, 100, 25, 100, 100, 100, 100, 100])
    # limit = torch.tensor([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 5, 5, 15, 10, 15, 10, 50, 25, 25, 25, 25, 5, 0, 15, 10, 15, 10, 50,
    #          25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 15, 10, 15, 10, 50, 25, 25, 25, 15, 10, 15, 10, 50, 25, 25, 25, 15,
    #          15, 50, 25, 25, 15, 0, 50, 25, 25, 25, 50, 25, 25, 50, 25, 25, 25, 50, 25, 25])
    length = []
    name = []
    act = torch.nn.Hardtanh(min_val= 0, max_val=1)
    Bone = {'R_shoulder' : [7,10],'R_up_arm': [10, 11], 'R_down_arm': [11, 12], 'L_shoulder' : [7,13],'L_up_arm': [13, 14], 'L_down_arm': [14, 15],
            'R_up_leg': [0, 1], 'R_down_leg': [1, 2], 'L_up_leg': [3, 4], 'L_down_leg': [4, 5], 'torso': [6, 7],
            'pelvis': [0, 3], 'head': [7, 9]}

    Bone2 = {'R_shoulder': [7, 10], 'R_up_arm': [10, 11], 'R_down_arm': [11, 12], 'L_shoulder': [7, 13],
            'L_up_arm': [13, 14], 'L_down_arm': [14, 15],
            'R_up_leg': [0, 1], 'R_down_leg': [1, 2], 'L_up_leg': [3, 4], 'L_down_leg': [4, 5], 'torso': [6, 7],
            'pelvis': [0, 3], 'head': [7, 9]}

    for j in Bone:
        Bone2.pop(j)
        for k in Bone2:
            segment1 = Bone[j]
            segment2 = Bone[k]
            if segment1[0] == segment2[0] or segment1[0] == segment2[1] or segment1[1] == segment2[0] or segment1[1] == segment2[1]:
                pass
            elif j == 'torso' and k == 'pelvis':
                pass
            elif j == 'R_down_arm' and k == 'L_down_arm':
                pass
            else:

                p1 = pose[:, :, segment1[0]]
                p2 = pose[:, :, segment1[1]]
                q1 = pose[:, :, segment2[0]]
                q2 = pose[:, :, segment2[1]]
                u = p2 - p1
                v = q2 - q1
                w0 = p1 - q1
                a = torch.bmm(u.reshape(-1,1,3), u.reshape(-1,3,1)).reshape(-1,1)
                b = torch.bmm(u.reshape(-1,1,3), v.reshape(-1,3,1)).reshape(-1,1)
                c = torch.bmm(v.reshape(-1,1,3), v.reshape(-1,3,1)).reshape(-1,1)
                d = torch.bmm(u.reshape(-1,1,3), w0.reshape(-1,3,1)).reshape(-1,1)
                e = torch.bmm(v.reshape(-1,1,3), w0.reshape(-1,3,1)).reshape(-1,1)
                f = torch.bmm(w0.reshape(-1, 1, 3), w0.reshape(-1, 3, 1)).reshape(-1, 1)


                length.append(torch.where((a*c - b*b).squeeze() == 0 , ((f - (e/c**0.5)**2) **0.5).squeeze() , torch.norm(((q1 + act((a*e - d*b) / (a*c - b*b)) * v)-(p1 + act((b*e - c*d) / (a*c - b*b)) * u)),p=2,dim=1)))
                name.append([j,k])
    length = torch.stack(length)
    # length = torch.where(length<30, 0, 1).reshape(-1,1)
    for a in range(length.shape[0]):

        if name[a][0][0] == "R" or name[a][0][0] == "L":
            b = eval(name[a][0][2:])
        else:
            b = eval(name[a][0])
        if name[a][1][0] == "R" or name[a][1][0] == "L":
            c = eval(name[a][1][2:])
        else:
            c = eval(name[a][1])
        limit = torch.stack([b,c]).max(0)[0]

        # print(name[a], limit.min(), length[a].max())
        length[a] = torch.where(length[a] <= limit, 0, 1)
    length = length.mean(0)
    # print(length)

    length = torch.where(length ==1, 1.,0.).reshape(-1,1)
    # print(length.sum()/length.shape[0])
    # exit()
    # print(length)
    # exit()
    return length

def cycle(input):
    while abs(input).max() > 1:
        input = torch.where(input > 1, 2 - input, input)
        input = torch.where(input < -1, -2 - input, input)

    return input

def freepose(batch_size, frame):
    rand_num = (torch.rand((batch_size, 34), device='cuda') * 2 - 1)
    ramdom_p3d = human_model3(rand_num)
    
    ramdom_p3d, ramdom_p2d, rot, trans = rand_position(ramdom_p3d)
    rand_tem = ((torch.randn((batch_size, 24), device='cuda')) / 10)

    tem_para = [rand_num * 1]
    tem_3d = [ramdom_p3d * 1]
    tem_2d = [ramdom_p2d * 1]

    for i in range(frame -1):
        rand_num[:, 8:32] += rand_tem
        rand_num[:, 8:32] = cycle(rand_num[:, 8:32])
        tem_para.append(rand_num * 1)
        ramdom_p3d = human_model3(rand_num)
        ramdom_p3d, ramdom_p2d, _, _ = rand_position(ramdom_p3d, rot, trans)
        tem_3d.append(ramdom_p3d)
        tem_2d.append(ramdom_p2d)
    
    tem_2d = torch.stack(tem_2d, dim=1).reshape(-1, frame * 32)
    p3d = tem_3d[int(frame/2)]
    distance = get_distance(p3d) 

    return p3d, tem_2d, distance