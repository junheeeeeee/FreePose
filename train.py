import torch
import torch.nn
import torch.optim
import numpy as np
import math
from torch.utils import data

from utils.data_val import H36MDataset_temp as ValDataset2
from utils.data_val import PW3DDataset_temp as ValDataset3
from utils.data_val import H36MDataset_temp as ValDataset4
import torch.optim as optim
import model_confidences
# from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng
import cv2
from utils.evaluate import PCK_3d, accuracy_3d, p_mpjpe, Metrics
# from utils.eval import p_mpjpe, get_pck3d, calc_auc, calc_auc_aligned
import os
# from utils.get_3d_skelton import get_3d_skeleton_for_canon as get_3d
# from torch.utils.tensorboard import SummaryWriter


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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

def get_distance(pose):

    # torso = torch.linalg.norm((pose[:,:,6] - pose[:,:,3]), dim=1) * 0.9
    # up_arm =  torch.linalg.norm((pose[:,:,13] - pose[:,:,14]), dim=1) / 16
    # down_arm = torch.linalg.norm((pose[:,:,14] - pose[:,:,15]), dim=1) / 16
    # up_leg = torch.linalg.norm((pose[:, :, 0] - pose[:, :, 1]), dim=1) / 16
    # down_leg = torch.linalg.norm((pose[:, :, 1] - pose[:, :, 2]), dim=1) / 16
    # a = torch.cross((pose[:, :, 7] - pose[:, :, 8]),(pose[:, :, 7] - pose[:, :, 9]))
    # head = torch.linalg.norm(a, dim=1) / torch.linalg.norm((pose[:, :, 7] - pose[:, :, 9]), dim=1) * 2
    # shoulder = up_arm * 1
    # pelvis = up_leg * 1
    # limit = torch.tensor(
    #     [25, 25, 25, 25, 25, 25, 25, 25, 100, 25, 5, 5, 15, 10, 15, 10, 100, 25, 100, 100, 25, 5, 0, 15, 10, 15, 10, 100,
    #      25, 100, 100, 25, 25, 25, 25, 25, 25, 100, 15, 10, 15, 10, 100, 25, 100, 100, 15, 10, 15, 10, 100, 25, 100, 100, 15,
    #      15, 100, 100, 100, 15, 0, 100, 25, 100, 100, 100, 100, 100, 100, 25, 100, 100, 100, 100, 100])
    # limit = torch.tensor([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 5, 5, 15, 10, 15, 10, 50, 25, 25, 25, 25, 5, 0, 15, 10, 15, 10, 50,
    #          25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 15, 10, 15, 10, 50, 25, 25, 25, 15, 10, 15, 10, 50, 25, 25, 25, 15,
    #          15, 50, 25, 25, 15, 0, 50, 25, 25, 25, 50, 25, 25, 50, 25, 25, 25, 50, 25, 25])
    length = []
    name = []
    # act = torch.nn.Hardtanh(min_val= 0, max_val=1)
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


config = SimpleNamespace()

config.learning_rate = 0.001
config.BATCH_SIZE = 1024
config.N_epochs = 800
# weights for the different losses
config.weight_rep = 2
config.weight_view = 0.05
config.weight_review = 2
config.weight_human = 0.01
config.data_size = 100
data_folder = './data/'
config.datafile = data_folder + 'pred_train.pkl'

config.valdatafile = data_folder + 'annot_test_full.pkl'

# loading the H36M dataset


val_dataset2 = ValDataset2(normalize_2d=True)
val_dataset3 = ValDataset3(normalize_2d=True)
val_dataset4 = ValDataset4(gt=False, normalize_2d=True)

val_loader2 = data.DataLoader(val_dataset2, batch_size=2048, shuffle=False, num_workers=6)
val_loader3 = data.DataLoader(val_dataset3, batch_size=2048, shuffle=False, num_workers=6)
val_loader4 = data.DataLoader(val_dataset4, batch_size=2048, shuffle=False, num_workers=6)


# load the skeleton morphing model as defined in Section 4.2
# for another joint detector it needs to be retrained -> train_skeleton_morph.py
def cycle(input):
    while abs(input).max() > 1:
        input = torch.where(input > 1, 2 - input, input)
        input = torch.where(input < -1, -2 - input, input)

    return input


def get_bone_lengths_all(poses):
    bone_map = [[6, 0], [0, 1], [1, 2], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 13], [13, 14],
                [14, 15], [7, 10], [10, 11], [11, 12]]

    poses = poses.reshape((-1, 3, 16))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


# loading the lifting network
model = model_confidences.Lifter_non_3d().cuda()

params = list(model.parameters())

optimizer = optim.Adam(params, lr=config.learning_rate)


losses = SimpleNamespace()
losses_mean = SimpleNamespace()

mse_loss = torch.nn.MSELoss(reduction='none')

best_H = 200
best_M = 200
best_3 = 200
best_Hp = 200
cc = 0
ac_2d = AverageMeter()
ac_3d = AverageMeter()
max_norm = 1
H36M_mpj3d = AverageMeter()


H36M_pck_3d = AverageMeter()
H36M_Ppck_3d = AverageMeter()
multi_3d = AverageMeter()

H36M_P_mpj3d = AverageMeter()


for epoch in range(config.N_epochs):

    for i in range(config.data_size):
        ii = epoch * config.data_size + i

        rand_num = (torch.rand((config.BATCH_SIZE, 34), device='cuda') * 2 - 1)


        rand_num_0 = rand_num * 1
        rand_num_1 = rand_num * 1
        rand_tem = ((torch.randn((config.BATCH_SIZE, 24), device='cuda')) / 10)
        rand_num_0[:, 8:32] += rand_tem
        rand_num_0[:, 8:32] = cycle(rand_num_0[:, 8:32])

        rand_tem = ((torch.randn((config.BATCH_SIZE, 24), device='cuda')) / 10)
        rand_num_1[:, 8:32] += rand_tem
        rand_num_1[:, 8:32] = cycle(rand_num_1[:, 8:32])


        ramdom_p3d = human_model(rand_num)
        distance = get_distance(ramdom_p3d)
        ramdom_p3d_0 = human_model(rand_num_0)

        ramdom_p3d_1 = human_model(rand_num_1)

        ramdom_p3d, ramdom_p2d, rot, trans = rand_position(ramdom_p3d)
        ramdom_p3d_0, ramdom_p2d_0, _, _ = rand_position(ramdom_p3d_0, rot, trans)
        ramdom_p3d_1, ramdom_p2d_1, _, _ = rand_position(ramdom_p3d_1, rot, trans)

        ramdom_p2d_temp = torch.stack([ramdom_p2d_0, ramdom_p2d, ramdom_p2d_1], dim=1).reshape(-1, 3 * 32)


        pred = model(ramdom_p2d_temp)

        losses.p3d = mse_loss( p3d_no_scale(pred[0].reshape(-1, 3, 16)) , p3d_no_scale(
            ramdom_p3d.reshape(-1, 3, 16) - ramdom_p3d.reshape(-1, 3, 16)[:, :, 6][:, :, None]))
        losses.p3d = (losses.p3d.sum(1) ** 0.5 * distance).mean()

        losses.loss = losses.p3d
        optimizer.zero_grad()
        losses.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        with torch.no_grad():

            annot_p3d = (ramdom_p3d.reshape(-1, 3, 16)).permute(0, 2, 1).detach().cpu().numpy()

            pred_p3d = pred[0].reshape(-1, 3, 16).permute(0, 2, 1).detach().cpu().numpy()
            pred_p3d -= pred_p3d.mean(1)[:, None, :]
            annot_p3d -= annot_p3d.mean(1)[:, None, :]

            pck, a3d = PCK_3d(pred_p3d, annot_p3d)
            ac_3d.update(a3d.mean(), ramdom_p3d.size(0))

        losses.mpj3d = ac_3d.avg



    ac_2d.reset()
    ac_3d.reset()


    with torch.no_grad():
        joint_mpj = []
        for i, sample in enumerate(val_loader2):
            # not the most elegant way to extract the dictionary

            inp_poses = sample[0].cuda().type(torch.float32)

            inp_poses = inp_poses.reshape(-1, 3 * 32)

            # predict 3d poses

            pred = model(inp_poses)
            # pred_poses = get_3d_point(pred[0])
            pred_poses = torch.transpose(pred[0].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            pred_poses = pred_poses - pred_poses.mean(1)[:, None, :]

            target = torch.transpose(sample[2].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            target = target - target.mean(1)[:, None, :]

            Ppck, pmpj = PCK_3d(pred_poses, target, alignment='procrustes')
            H36M_P_mpj3d.update(pmpj.mean(), inp_poses.size(0))
            pckk, mpj = PCK_3d(pred_poses, target)
            joint_mpj.append(mpj)
            H36M_mpj3d.update(mpj.mean(), inp_poses.size(0))
            H36M_pck_3d.update(pckk, inp_poses.size(0))

    msg = 'H36M :\t' \
          'n-mpjpe {acc.avg:.1f}mm\t' \
          'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)
    print(msg)
    joint = np.concatenate(joint_mpj, axis=0).mean(0)
    # print(f' hip : {joint[6]}, r_hip : {joint[0]}, r_knee : {joint[2]}, r_foot : {joint[3]}, neck : {joint[8]}, nosie : {joint[9]}, head : {joint[10]}, l_shoulder : {joint[11]}, l_elbow : {joint[12]}, l_wrist : {joint[13]}')
    # save the new trained model every epoch
    # torch.save(model, 'models/model_lifter_single.pt')
    if best_H > H36M_mpj3d.avg:
        if os.path.isfile(f'output/model_lifter_single_H36M_{best_H}.pt'):
            os.remove(f'output/model_lifter_single_H36M_{best_H}.pt')
        best_H = H36M_mpj3d.avg
        torch.save(model, f'output/model_lifter_single_H36M_{best_H}.pt')
        print('save_best')
        cc = 0

    H36M_mpj3d.reset()
    H36M_P_mpj3d.reset()
    H36M_pck_3d.reset()

    with torch.no_grad():
        for i, sample in enumerate(val_loader4):
            # not the most elegant way to extract the dictionary

            inp_poses = sample[0].cuda().type(torch.float32)

            inp_poses = inp_poses.reshape(-1, 3 * 32)

            # predict 3d poses

            pred = model(inp_poses)
            # pred_poses = get_3d_point(pred[0])
            pred_poses = torch.transpose(pred[0].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            pred_poses = pred_poses - pred_poses.mean(1)[:, None, :]

            target = torch.transpose(sample[2].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            target = target - target.mean(1)[:, None, :]

            Ppck, pmpj = PCK_3d(pred_poses, target, alignment='procrustes')
            H36M_P_mpj3d.update(pmpj.mean(), inp_poses.size(0))
            pckk, mpj = PCK_3d(pred_poses, target)
            H36M_mpj3d.update(mpj.mean(), inp_poses.size(0))
            H36M_pck_3d.update(pckk, inp_poses.size(0))

    msg = 'H36Mp :\t' \
          'n-mpjpe {acc.avg:.1f}mm\t' \
          'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)

    print(msg)

    # save the new trained model every epoch
    # torch.save(model, 'models/model_lifter_single.pt')
    if best_Hp > H36M_mpj3d.avg:
        if os.path.isfile(f'output/model_lifter_single_H36Mpred_{best_Hp}.pt'):
            os.remove(f'output/model_lifter_single_H36Mpred_{best_Hp}.pt')
        best_Hp = H36M_mpj3d.avg
        torch.save(model, f'output/model_lifter_single_H36Mpred_{best_Hp}.pt')
        print('save_best')
        cc = 0
    # if not (epoch+1) % 30:
    #     scheduler.step()
    # scheduler.step(H36M_mpj3d.avg)
    H36M_mpj3d.reset()
    H36M_P_mpj3d.reset()
    H36M_pck_3d.reset()
    with torch.no_grad():
        for i, sample in enumerate(val_loader3):
            # not the most elegant way to extract the dictionary

            inp_poses = sample[0].cuda().type(torch.float32)

            inp_poses = inp_poses.reshape(-1, 3 * 32)

            # predict 3d poses

            pred = model(inp_poses)
            # pred_poses = get_3d_point(pred[0])
            pred_poses = torch.transpose(pred[0].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            pred_poses = pred_poses - pred_poses.mean(1)[:, None, :]

            target = torch.transpose(sample[2].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            target = target - target.mean(1)[:, None, :]

            Ppck, pmpj = PCK_3d(pred_poses, target, alignment='procrustes')
            H36M_P_mpj3d.update(pmpj.mean(), inp_poses.size(0))
            pckk, mpj = PCK_3d(pred_poses, target)
            H36M_mpj3d.update(mpj.mean(), inp_poses.size(0))
            H36M_pck_3d.update(pckk, inp_poses.size(0))

    msg = '3DPW :\t' \
          'n-mpjpe {acc.avg:.1f}mm\t' \
          'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)

    print(msg)


    if best_3 > H36M_mpj3d.avg:
        if os.path.isfile(f'output/model_lifter_single_3DPW_{best_3}.pt'):
            os.remove(f'output/model_lifter_single_3DPW_{best_3}.pt')
        best_3 = H36M_mpj3d.avg
        torch.save(model, f'output/model_lifter_single_3DPW_{best_3}.pt')
        print('save_best')
        cc = 0

    print()
    print(f'Best')
    print(f'H : {best_H} Hp : {best_Hp} 3dpw : {best_3}' )
    H36M_mpj3d.reset()
    H36M_P_mpj3d.reset()
    H36M_pck_3d.reset()
    print()
print('done')
