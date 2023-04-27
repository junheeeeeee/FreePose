import torch
import torch.nn as nn
from pytorch3d.transforms import so3_exponential_map as rodrigues

class Lifter(nn.Module):
    def __init__(self):
        super(Lifter, self).__init__()

        self.upscale = nn.Linear(32+ 16, 1024)
        self.res_common = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.res_cam1 = res_block()
        self.res_cam2 = res_block()
        self.pose3d = nn.Linear(1024, 48)
        self.enc_rot = nn.Linear(1024, 3)

    def forward(self, p2d,conf):

        x = torch.cat((p2d, conf), axis=1)


        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        z_pose = self.pose3d(xp)
        z_pose = z_pose.view(-1, 3, 16)
        z_pose[:, :, 6] = torch.zeros_like(z_pose[:, :, 6])
        z_pose[:, 2, 0] = torch.zeros_like(z_pose[:, 2, 0])
        z_pose[:, 2, 3] = torch.zeros_like(z_pose[:, 2, 3])
        z_pose[:, 2, 7] = torch.zeros_like(z_pose[:, 2, 7])
        # camera path
        xc = nn.LeakyReLU()(self.res_cam1(x))
        xc = nn.LeakyReLU()(self.res_cam2(xc))
        xc = self.enc_rot(xc)

        return z_pose, xc


class Lifter_non(nn.Module):
    def __init__(self):
        super(Lifter_non, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()

        self.pose3d = nn.Linear(1024, 48)


    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.view(-1, 48) , p2d

class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.du1 = nn.Dropout(p = 0.25)
        self.du2 = nn.Dropout(p=0.25)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        # x = self.bn1(x)
        x = nn.LeakyReLU()(self.l2(x))
        x = self.bn2(x)
        x += inp

        return x

class Lifter_non_unbox(nn.Module):
    def __init__(self):
        super(Lifter_non_unbox, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()

        self.pose3d = nn.Linear(1024, 48)


    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        z_pose = self.pose3d(xp)
        z_pose = z_pose.view(-1,3,16)
        z_pose[:,2] = nn.ReLU()(z_pose[:,2]) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,0] * -1 + z_pose[:,:,6] * 2



        return z_pose.view(-1, 48) , p2d

class Lifter_non_2d(nn.Module):
    def __init__(self):
        super(Lifter_non_2d, self).__init__()

        self.upscale = nn.Linear(32*3, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()

        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()
        self.trans = nn.Linear(1024, 1)

        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 16)



    def forward(self, p2d):


        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        deep = self.pose3d(xp)


        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))
        trans = nn.ReLU()(self.trans(xt)) * 100 + 900

        # xr = nn.LeakyReLU()(self.rot_pose1(x))
        # xr = nn.LeakyReLU()(self.rot_pose2(xr))
        # rot = rodrigues(self.rot(xr))

        z_pose = deep.unsqueeze(1) + trans.unsqueeze(1)
        pose = torch.cat((p2d[:,32:64].reshape(-1,2,16)*z_pose ,z_pose),dim=1)


        return pose , deep , trans

class Lifter_human(nn.Module):
    def __init__(self):
        super(Lifter_human, self).__init__()

        self.upscale = nn.Linear(32*3, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()


        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()
        self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 32)



    def forward(self, p2d):


        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        human = torch.sin(self.pose3d(xp))


        # xt = nn.LeakyReLU()(self.trans_pose1(x))
        # xt = nn.LeakyReLU()(self.trans_pose2(xt))
        # trans = nn.ReLU()(self.trans(xt)) * 100 + 900

        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))
        rot = rodrigues(self.rot(xr))
        pose = rot @ human_model(human)

        return pose, human , rot

class Lifter_video(nn.Module):
    def __init__(self):
        super(Lifter_video, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()
        self.downscale = nn.Linear(1024 * 3, 1024)
        self.res_pose3 = res_block()

        self.res_pose4 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48)



    def forward(self, p2d):
        p2d = p2d.reshape(-1,32)


        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        xp = xp.reshape(-1,1024 * 3)

        xp = self.downscale(xp)
        xp = nn.LeakyReLU()(self.res_pose3(xp))
        xp = nn.LeakyReLU()(self.res_pose4(xp))
        pose = self.pose3d(xp)






        return [pose , pose]

class Lifter_non_3d(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.n_joints = 16

        self.upscale = nn.Linear(frame * 2 * self.n_joints, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48)



    def forward(self, p2d):

        
        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        pose = self.pose3d(xp)






        return [pose , pose]

class Lifter_non_3d2(nn.Module):
    def __init__(self):
        super(Lifter_non_3d2, self).__init__()

        self.upscale = nn.Linear(32*3, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48)



    def forward(self, p2d):

        p2d = p2d.reshape(-1,2,16)
        p2d -= p2d[:,:,6][:,:,None]

        p2d /= (torch.linalg.norm(p2d, dim=1).mean(-1))[:,None,None]
        p2d = p2d.reshape(-1,3*2*16)
        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        pose = self.pose3d(xp)






        return [pose , pose]

class Lifter_non_3d_frame(nn.Module):
    def __init__(self, frame):
        super(Lifter_non_3d_frame, self).__init__()

        self.upscale = nn.Linear(32*frame, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48)



    def forward(self, p2d):


        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        pose = self.pose3d(xp)






        return [pose , pose]

class Lifter_non_3d_3(nn.Module):
    def __init__(self):
        super(Lifter_non_3d_3, self).__init__()

        self.upscale = nn.Linear(32*3, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        # self.res_pose3 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48*3)



    def forward(self, p2d):


        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        pose = self.pose3d(xp)

        pose = pose.reshape(-1,3,48)




        return [pose[:,1] , pose.reshape(-1,48)]

class Lifter_non_3d_T(nn.Module):
    def __init__(self):
        super(Lifter_non_3d_T, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()

        self.downscale = nn.Linear(3072, 1024)
        self.res_pose3 = res_block()

        self.res_pose4 = res_block()
        self.res_pose5 = res_block()

        # self.res_pose3 = res_block()


        # self.rot_pose1 = res_block()
        # self.rot_pose2 = res_block()
        # self.rot = nn.Linear(1024, 3)

        self.pose3d = nn.Linear(1024, 48)



    def forward(self, p2d):
        temp = []
        xp = []
        temp.append(p2d[:, :32])
        temp.append(p2d[:, 32:64])
        temp.append(p2d[:, 64:])
        for i in temp:
            x = self.upscale(i)
            x = nn.LeakyReLU()(self.res_pose0(x))

            # pose path
            x = nn.LeakyReLU()(self.res_pose1(x))
            x = nn.LeakyReLU()(self.res_pose2(x))
            xp.append(x)
        xp = torch.cat(xp,dim=-1)
        xp = self.downscale(xp)
        xp = nn.LeakyReLU()(self.res_pose3(xp))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose4(xp))
        xp = nn.LeakyReLU()(self.res_pose5(xp))

        pose = self.pose3d(xp)

        return [pose , pose]
class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x

def human_model(pred):

    ## base human
    pred = pred.unsqueeze(-1)
    # pred[:, :8]  =  pred[:, :8] * 0

    p = torch.zeros((pred.shape[0],17,3)).cuda()
    x = torch.zeros((pred.shape[0],3)).cuda() + torch.tensor([1,0,0]).cuda()
    y = torch.zeros((pred.shape[0],3)).cuda() + torch.tensor([0,1,0]).cuda()
    z = torch.zeros((pred.shape[0],3)).cuda() + torch.tensor([0,0,1]).cuda()

    p[:, 1] = p[:, 1] - x * 265.9/2 * (1 + pred[:,0]/3)
    p[:, 4] = p[:, 4] + x * 265.9/2 * (1 + pred[:,0]/3)

    p[:, 8] = p[:, 8] + z * 488.9 * (1 + pred[:,1]/3)
    p[:, 10] = p[:, 8] + z * 187.6 * (1 + pred[:,1]/3)
    p[:, 9] = (p[:, 8]+p[:, 10])/2 + y * 70 * (1 + pred[:,2]/3)

    head_rot = rodrigues(z * ((pred[:, 8]) * 45) * torch.pi / 180)
    head_front = rodrigues(-x * (pred[:, 9] * 60 + 10) * torch.pi / 180)
    head_side = rodrigues(y * ((pred[:, 10]) * 35) * torch.pi / 180)
    p[:, 9:11] = torch.transpose(head_front @ head_side @ head_rot @ torch.transpose(p[:,9:11]-p[:,8][:,None,:],1,2) ,1,2) + p[:,8][:,None,:]

    shod_l_rot = rodrigues(y * torch.arcsin((pred[:, 11] + 1) * 0.8 / 2))
    shod_len = (x * 151 * (1 + pred[:, 3] / 3))[:, :, None]
    shod_l = shod_l_rot.matmul(shod_len).squeeze(-1)
    p[:, 11] = p[:, 8] + shod_l
    shod_r_rot = rodrigues(-y * torch.arcsin((pred[:, 12] + 1) * 0.8 / 2))
    shod_r = shod_r_rot.matmul(shod_len).squeeze(-1)
    p[:, 14] = p[:, 8] - shod_r

    p[:, 12] = p[:, 11] - z * 278.9 * (1 + pred[:, 4] / 3)
    p[:, 15] = p[:, 14] - z * 278.9 * (1 + pred[:, 4] / 3)

    elbow_l_rot = rodrigues(x * (pred[:, 13] + 1) / 2 * 3.14 * 145 / 180)

    elbow_l_rot2 = rodrigues(z * ((pred[:, 14] + 1) / 2 * 130 - 20) * 3.14 / 180)

    elbow_l = (elbow_l_rot2 @ elbow_l_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 13] = p[:, 12] - elbow_l * 251.7 * (1 + pred[:, 5] / 3)

    side_angel = ((pred[:, 15] + 1) / 2 * 140 - 90)
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

    elbow_r_rot2 = rodrigues(-z * ((pred[:, 18] + 1) / 2 * 130 - 20) * 3.14 / 180)

    elbow_r = (elbow_r_rot2 @ elbow_r_rot).matmul(z[:, :, None]).squeeze(-1)
    p[:, 16] = p[:, 15] - elbow_r * 251.7 * (1 + pred[:, 5] / 3)

    side_angel = ((pred[:, 19] + 1) / 2 * 140 - 90)
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
    p[:, 2] = p[:, 1] - z * 442.9 * (1 + pred[:,6]/3)
    p[:, 5] = p[:, 4] - z * 442.9 * (1 + pred[:,6]/3)
    knee_r_rot = rodrigues(-x * (pred[:, 21] + 1) / 2 * torch.pi * 135 / 180)
    knee_r_rot2 = rodrigues(-z * (pred[:, 22] * 45) * torch.pi / 180)
    knee_r = knee_r_rot2.matmul(knee_r_rot.matmul(z[:,:,None])).squeeze(-1)
    p[:, 3] = p[:, 2] - knee_r * 454.2 * (1 + pred[:,7]/3)
    knee_r_side = rodrigues(y * ((pred[:, 23] + 1) / 2 * 70 - 25) * torch.pi / 180)
    knee_r_front = rodrigues(x * ((pred[:, 24] + 1) / 2 * 135 - 45) * torch.pi / 180)
    p[:, 1:4] = torch.transpose(knee_r_front @ knee_r_side @ torch.transpose(p[:, 1:4] - p[:, 1][:, None, :], 1, 2), 1,
                                2) + p[:, 1][:, None, :]

    knee_l_rot = rodrigues(-x * (pred[:, 25] + 1) / 2 * torch.pi * 135 / 180)
    knee_l_rot2 = rodrigues(z * (pred[:, 26] * 45) * torch.pi / 180)
    knee_l = knee_l_rot2.matmul(knee_l_rot.matmul(z[:,:,None])).squeeze(-1)
    p[:, 6] = p[:, 5] - knee_l * 454.2 * (1 + pred[:,7]/3)
    knee_l_side = rodrigues(-y * ((pred[:, 27] + 1) / 2 * 70 - 25) * torch.pi / 180)
    knee_l_front = rodrigues(x * ((pred[:, 28] + 1) / 2 * 135 - 45) * torch.pi / 180)
    p[:, 5:7] = torch.transpose(knee_l_front @ knee_l_side @ torch.transpose(p[:, 5:7] - p[:, 4][:, None, :], 1, 2), 1,
                                2) + p[:, 4][:, None, :]

    spine_rot = rodrigues(z * ((pred[:, 29]) * 30) * torch.pi / 180)
    spine_front = rodrigues(-x * ((pred[:, 30] + 1) / 2 * 105 - 30) * torch.pi / 180)
    spine_side = rodrigues(y * ((pred[:, 31]) * 35) * torch.pi / 180)
    p[:, 7:] = torch.transpose(spine_front @ spine_side @ spine_rot @ torch.transpose(p[:,7:] * 1,1,2) ,1,2)
    # p[:, :7] = torch.transpose(spine_front @ torch.transpose(p[:, :7] * 1, 1, 2), 1, 2)
    index = [1,2,3,4,5,6,0,8,9,10,14,15,16,11,12,13]
    pp = torch.zeros((pred.shape[0],16,3)).cuda()
    for i, j in enumerate(index):
        pp[:, i] = p[:, j]
    pp = torch.transpose(pp,1,2).reshape(-1,3,16)
    p3d = torch.zeros_like(pp)
    p3d[:,0] = pp[:,0]
    p3d[:, 1] = -pp[:, 2]
    p3d[:, 2] = -pp[:,1]
    return p3d

def p3d_no_scale(p3d):
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
    scale_p3d = bone_lenth.mean(-1)[:,None,None]

    p3d_scaled = p3d/scale_p3d

    # loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return p3d_scaled

class Lifter_single(nn.Module):
    def __init__(self):
        super(Lifter_single, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.res_pose3 = res_block()

        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()
        self.rot_pose3 = res_block()

        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()

        self.human = nn.Linear(1024, 30)

        self.rot = nn.Linear(1024, 3)

        self.trans = nn.Linear(1024, 3)

    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        xp = nn.LeakyReLU()(self.res_pose3(xp))

        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))
        xr = nn.LeakyReLU()(self.rot_pose3(xr))
        xr = self.rot(xr)

        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))



        rot = rodrigues(xr)

        trans = self.trans(xt)
        xpp = nn.Hardsigmoid()(self.human(xp))*2.5 - 1.25
        human = p3d_no_scale(human_model(xpp))



        z_pose =(rot @ human ) + trans[:,:,None]
        z_pose[:, 2] = z_pose[:, 2] + 12
        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.reshape(-1, 48) , xpp , xr

class Lifter_single_temp(nn.Module):
    def __init__(self):
        super(Lifter_single_temp, self).__init__()

        self.upscale = nn.Linear(32 * 3, 1024)
        self.res_pose0 = res_block()


        self.res_pose1 = res_block()
        self.res_pose2 = res_block()


        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()

        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()

        self.human = nn.Linear(1024, 30)

        self.rot = nn.Linear(1024, 3)

        self.trans = nn.Linear(1024, 3)

    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))


        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))
        xr = self.rot(xr)

        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))



        rot = rodrigues(xr)

        trans = self.trans(xt)
        xpp = nn.Hardsigmoid()(self.human(xp))*2.5 - 1.25
        human = p3d_no_scale(human_model(xpp))



        z_pose =(rot @ human ) + trans[:,:,None]
        z_pose[:, 2] = z_pose[:, 2] + 12
        z_pose[:, 2] = nn.ReLU()(z_pose[:, 2] - 1) + 1
        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.reshape(-1, 48) , xpp , xr

class Lifter_single_temp_cos(nn.Module):
    def __init__(self):
        super(Lifter_single_temp_cos, self).__init__()

        self.upscale = nn.Linear(32 * 3, 1024)
        self.res_pose0 = res_block()

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()


        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()


        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()

        self.human = nn.Linear(1024, 30)

        self.rot = nn.Linear(1024, 3)

        self.trans = nn.Linear(1024, 3)

    def forward(self, p2d):



        x = self.upscale(p2d)

        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))


        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))
        xr = self.rot(xr)

        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))



        rot = rodrigues(xr)

        trans = self.trans(xt)
        # xpp = torch.sin(self.human(xp/10))*1.05
        xpp = torch.nn.Hardtanh()(self.human(xp / 10)) * 1.05
        human = p3d_no_scale(human_model(xpp))
        trans_f = trans.unsqueeze(-1) * 1
        trans_f[:, 0] = (trans_f[:, 0] * 2048 - 1024) * 0.9
        trans_f[:, 1] = (trans_f[:, 1] * 2048 - 1024) * 0.9
        trans_f[:, 2] = trans_f[:, 2] * 5600 + 900


        z_pose =(rot @ human ) + trans_f
        z_pose[:, 2] = z_pose[:, 2] + 12
        z_pose[:, 2] = nn.ReLU()(z_pose[:, 2] - 0.1) + 0.1
        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.reshape(-1, 48) , xpp , rot ,trans

class Lifter_single_temp_cos_unbox(nn.Module):
    def __init__(self):
        super(Lifter_single_temp_cos_unbox, self).__init__()

        self.upscale = nn.Linear(32 * 3, 1024)
        self.res_pose0 = res_block()
        self.pose = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()


        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()


        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()

        self.human = nn.Linear(1024, 32)

        self.rot = nn.Linear(1024, 3)

        self.trans = nn.Linear(1024, 3)

    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.pose(x))
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))


        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))

        xr = self.rot(xr)

        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))


        rot = rodrigues(xr)

        trans = torch.nn.Hardsigmoid()(self.trans(xt/5)) * 1.05
        trans_point = trans * 1
        trans[:, 0] = (trans[:, 0] * 2048 - 1024)
        trans[:, 1] = (trans[:, 1] * 2048 - 1024)
        trans[:, 2] = (trans[:, 2] * 5600 + 900)
        xpp = torch.nn.Hardtanh()(self.human(xp/10))*1.05
        human = human_model(xpp)



        z_pose =(rot @ human ) + trans[:,:,None]

        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.reshape(-1, 48) , xpp , rot, trans_point

class Lifter_single_sin(nn.Module):
    def __init__(self):
        super(Lifter_single_sin, self).__init__()

        self.upscale = nn.Linear(32, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()

        self.rot_pose1 = res_block()
        self.rot_pose2 = res_block()

        self.trans_pose1 = res_block()
        self.trans_pose2 = res_block()

        self.human = nn.Linear(1024, 32)

        self.rot = nn.Linear(1024, 3)

        self.trans = nn.Linear(1024, 3)

    def forward(self, p2d):



        x = self.upscale(p2d)
        x = nn.LeakyReLU()(self.res_pose0(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))

        xr = nn.LeakyReLU()(self.rot_pose1(x))
        xr = nn.LeakyReLU()(self.rot_pose2(xr))

        xt = nn.LeakyReLU()(self.trans_pose1(x))
        xt = nn.LeakyReLU()(self.trans_pose2(xt))



        rot = rodrigues(self.rot(xr))

        trans = self.trans(xt)
        xpp = torch.sin(self.human(xp)/2)
        human = p3d_no_scale(human_model(xpp))



        z_pose =(rot @ human ) + trans[:,:,None]
        z_pose[:, 2] = z_pose[:, 2] + 12
        # z_pose = z_pose.view(-1,3,16)
        # z_pose[:,2] = nn.ReLU()(z_pose[:,2] + 10) + 1
        # z_pose[:,:,6] = torch.zeros_like(z_pose[:,:,6])
        # z_pose[:,:,3] = z_pose[:,:,6] *2 - z_pose[:,:,0]



        return z_pose.reshape(-1, 48) , xpp , human

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x



class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=8, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3+16, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):


        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet_Transform(nn.Module):
    def __init__(self, block, num_block, num_classes=8, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.relationship = torch.nn.parameter.Parameter(data=torch.eye(16)[None,:,:], requires_grad=True)


        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sm = nn.Softmax(dim=-1)
        # weights inittialization
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        img = x[:,:3]

        heat = (self.relationship @ x[:, 3:].reshape(-1,16,256*256)).reshape(-1,16,256,256)

        img = self.conv(img)

        img = heat * img + img

        output = self.conv1(img)


        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)


        x = self.conv5_x(x)
        f_shape = x.shape

        x = self.avg_pool(x)
        x = x.view(f_shape[0], -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet_Transform2(nn.Module):
    def __init__(self, block, num_block, num_classes=8, init_weights=True):
        super().__init__()

        self.in_channels=64


        self.conv_Q2 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.conv_K2 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv_V2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.conv_A = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.conv_im = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.conv_ht = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.conv_fs = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sm = nn.Softmax(dim=-1)
        # weights inittialization
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):

        img = x[:, :3]

        heat = self.conv_ht(x[:, 3:])

        img = self.conv_im(img)
        fusion = self.conv_fs(heat * img)
        output = fusion + img

        output = self.conv1(output)


        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        f_shape = x.shape

        Quary = self.conv_Q2(x[:, int(f_shape[1]/2):int(f_shape[1]/2 + f_shape[1]/4)]).reshape(-1, f_shape[1], f_shape[2] * f_shape[3]).permute(0, 2, 1)

        Key = self.conv_K2(x[:, int(f_shape[1]/2 + f_shape[1]/4):]).reshape(-1, f_shape[1], f_shape[2] * f_shape[3])

        Value = self.conv_V2(x[:, :int(f_shape[1]/2)]).reshape(-1, f_shape[1], f_shape[2] * f_shape[3])

        Attention = self.conv_A((Value @ (self.sm(Quary @ Key))).reshape(f_shape))
        x = x + Attention

        x = self.conv5_x(x)



        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        # exit()
        x = x.view(f_shape[0], -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50(a):
    return ResNet(BottleNeck, [3,4,6,3],num_classes=a)
def resnet50_t(a):
    return ResNet_Transform(BottleNeck, [3,4,6,3],num_classes=a)
def resnet50_t2(a):
    return ResNet_Transform2(BottleNeck, [3,4,6,3],num_classes=a)


def resnet101(a):
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=a)
def resnet101_t(a):
    return ResNet_Transform(BottleNeck, [3, 4, 23, 3],num_classes=a)
def resnet101_t2(a):
    return ResNet_Transform2(BottleNeck, [3, 4, 23, 3],num_classes=a)

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

class deep_model(nn.Module):
    def __init__(self):
        super(deep_model, self).__init__()
        self.upscale = nn.Linear(64, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet50(32)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)


        depth_inf = self.encoder(x)

        x = torch.cat(( p2d,depth_inf), dim=-1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_pose0(x))
        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        z_pose = self.pose3d(xp)
        z_pose = z_pose.view(-1,3,16)

        return z_pose, p2d

class deep_model_ver2(nn.Module):
    def __init__(self):
        super(deep_model_ver2, self).__init__()
        self.upscale = nn.Linear(48, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet50(16)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)

        depth_inf = self.encoder(x)
        x = torch.cat((p2d, depth_inf), dim=-1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_pose0(x))
        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        z_pose = self.pose3d(xp)
        z_pose = z_pose.view(-1,3,16)

        return z_pose, p2d

class deep_model_z(nn.Module):
    def __init__(self):
        super(deep_model_z, self).__init__()
        self.upscale = nn.Linear(48, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()


        self.rz_pose1 = res_block()
        self.rz_pose2 = res_block()

        self.pose3d = nn.Linear(1024, 48)
        self.root_z = nn.Linear(1024, 1)

        self.encoder = resnet50(16)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)

        depth_inf = self.encoder(x)
        x = torch.cat((p2d, depth_inf), dim=-1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_pose0(x))
        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        zp = nn.LeakyReLU()(self.res_pose1(x))
        zp = nn.LeakyReLU()(self.res_pose2(zp))

        z_pose = self.pose3d(xp)
        root_z = nn.ReLU()(self.root_z(zp).reshape(-1,1)) * 10

        z_pose = z_pose.view(-1,3,16)

        return z_pose, root_z

class deep_model_test(nn.Module):
    def __init__(self):
        super(deep_model_test, self).__init__()
        self.upscale = nn.Linear(64, 1024)
        self.res_pose0 = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet101(48)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)


        depth_inf = self.encoder(x)

        # x = torch.cat(( p2d,depth_inf), dim=-1)
        x = self.upscale(depth_inf)
        x = nn.LeakyReLU()(self.res_pose0(x))
        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        # xp = nn.LeakyReLU()(self.res_pose3(xp))

        z_pose = self.pose3d(xp)
        z_pose = z_pose.view(-1,3,16)

        return z_pose, p2d

class deep_model_transform(nn.Module):
    def __init__(self):
        super(deep_model_transform, self).__init__()
        # self.upscale = nn.Linear(64, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet101_t(48)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)


        depth_inf = self.encoder(x)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return depth_inf, p2d

class deep_model_transform2(nn.Module):
    def __init__(self):
        super(deep_model_transform2, self).__init__()
        # self.upscale = nn.Linear(2048, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet101_t2(48)

    def forward(self, img, heatmap, p2d):

        x = torch.cat((img,heatmap),dim=1)


        depth_inf = self.encoder(x)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return depth_inf, p2d


class deep_model_transform2_50(nn.Module):
    def __init__(self):
        super(deep_model_transform2_50, self).__init__()
        # self.upscale = nn.Linear(2048, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet50_t2(48)

    def forward(self, img, heatmap, p2d):
        x = torch.cat((img, heatmap), dim=1)

        depth_inf = self.encoder(x)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return depth_inf, p2d

class deep_model_transform2_101(nn.Module):
    def __init__(self):
        super(deep_model_transform2_101, self).__init__()
        # self.upscale = nn.Linear(2048, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet101_t2(48)

    def forward(self, img, heatmap, p2d):
        x = torch.cat((img, heatmap), dim=1)

        depth_inf = self.encoder(x)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return depth_inf, p2d

class deep_model_transform3_50(nn.Module):
    def __init__(self):
        super(deep_model_transform3_50, self).__init__()
        # self.upscale = nn.Linear(2048, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet50_t2(16)


    def forward(self, img, heatmap, p2d):
        x = torch.cat((img, heatmap), dim=1)

        depth_inf = self.encoder(x) + 10
        depth_inf = nn.ReLU()(depth_inf - 1) + 1
        p2d = (p2d.reshape(-1,2,16) * depth_inf.unsqueeze(1)).reshape(-1,32)

        p3d = torch.cat([p2d, depth_inf],dim=-1)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return p3d, p2d

class deep_model_transform3_100(nn.Module):
    def __init__(self):
        super(deep_model_transform3_100, self).__init__()
        # self.upscale = nn.Linear(2048, 1024)
        # self.res_pose0 = res_block()
        # self.res_pose1 = res_block()
        # self.res_pose2 = res_block()
        # self.pose3d = nn.Linear(1024, 48)
        self.encoder = resnet101_t2(16)


    def forward(self, img, heatmap, p2d):
        x = torch.cat((img, heatmap), dim=1)

        depth_inf = self.encoder(x) + 10
        depth_inf = nn.ReLU()(depth_inf - 1) + 1
        p2d = (p2d.reshape(-1,2,16) * depth_inf.unsqueeze(1)).reshape(-1,32)

        p3d = torch.cat([p2d, depth_inf],dim=-1)

        # # x = torch.cat(( p2d,depth_inf), dim=-1)
        # x = self.upscale(depth_inf)
        # x = nn.LeakyReLU()(self.res_pose0(x))
        # # pose path
        # xp = nn.LeakyReLU()(self.res_pose1(x))
        # xp = nn.LeakyReLU()(self.res_pose2(xp))
        # # xp = nn.LeakyReLU()(self.res_pose3(xp))
        #
        # z_pose = self.pose3d(xp)
        # z_pose = z_pose.view(-1,3,16)

        return p3d, p2d