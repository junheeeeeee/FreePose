import torch
import torch.nn
import torch.optim
import numpy as np
import math
from torch.utils import data
from utils.utils import cycle, p3d_no_scale, rand_position, human_model3, get_distance, freepose
from utils.data_val import H36MDataset_temp as ValDataset2
from utils.data_val import PW3DDataset_temp as ValDataset3
from utils.data_val import H36MDataset_temp as ValDataset4
import torch.optim as optim
import model_confidences
from types import SimpleNamespace
from numpy.random import default_rng
import cv2
from utils.evaluate import PCK_3d, accuracy_3d, p_mpjpe, Metrics

import os



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

        ramdom_p3d, ramdom_p2d_temp, distance = freepose(config.BATCH_SIZE, 3)
     
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
