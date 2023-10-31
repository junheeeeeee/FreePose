import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
import math
import os
import cv2
import torchvision.transforms as transforms


def get_temp_frame(num, data_p2d, data_p3d):
    output = [[], []]
    set = ['S9', "S11"]
    index = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    for i in set:
        for j in data_p2d[i].keys():
            for k in range(4):
                for l in range(len(data_p2d[i][j][k])):
                    l += 1
                    if l % 64 == 0:
                        temp = []
                        for n in range(num):
                            try:
                                temp.append(data_p2d[i][j][k][l + n - 2][index])
                            except:
                                temp.append(data_p2d[i][j][k][l - 1][index])
                        output[0].append(np.stack(temp, axis=0))
                        output[1].append(data_p3d[i][j][k][l - 1])
    output[0] = np.array(output[0])
    output[1] = np.array(output[1])[:, index]
    return output

def get_temp_frame_my(num, data):
    output = [[], []]
    p2d = data[0]
    p3d = data[1]
    id = data[2]
    joint = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    for i in range(len(p2d)):
        id_t = id[i]
        temp = []
        for j in range(num):
            while True:
                index = i - int(num / 2) + j
                if index >= len(p2d):
                    index = len(p2d) - 1
                else:
                    pass
                if id_t == id[index]:
                    temp.append(p2d[index])
                    break
                else:
                    if i > index:
                        j += 1
                    else:
                        j -= 1
        output[0].append(np.stack(temp, axis=0))
        output[1].append(p3d[i])
    output[0] = np.array(output[0])[:, :, joint]
    output[1] = np.array(output[1])[:, joint]
    return output

class SkyDataset_temp(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'

        fname = data_folder + 'ski_test.pkl'


        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)


        # index = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]
        self.data = get_temp_frame_my(3, self.data)
        self.data[0] = np.transpose(self.data[0],(0,1,3,2))
        self.data[1] = np.transpose(self.data[1], (0, 2, 1))
        if normalize_2d:
            self.data[0] -= 128
            self.data[0] /= 512


        self.data[1] = self.data[1].reshape(-1,48)
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.data[0][idx]
        # p2d = self.data[0][idx]
        p3d = self.data[1][idx]
        confidences = np.ones(16)

        sample = [p2d,confidences,p3d]

        return sample

class H36MDataset(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, gt = True, normalize_2d=True):
        data_folder = './data/'
        if gt:
            fname = data_folder + 'h36m_gt_test.pkl'
        else:
            fname = data_folder + 'h36m_hr_test.pkl'

        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)


        if normalize_2d:
            self.data[0] = (self.data[0] - self.data[0][:, :, 6][:,:,None]).reshape(-1, 32)
            self.data[0] /= np.linalg.norm(self.data[0], ord=2, axis=1, keepdims=True)
        self.data[1] = self.data[1].reshape(-1,48)
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.data[0][idx]
        p3d = self.data[1][idx]
        confidences = np.ones(16)

        sample = [p2d,confidences,p3d]

        return sample

class MPI_3DHP(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot2.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 128
            self.data[0] /= 128
        self.data[1] = self.data[1].reshape(-1,48)


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.pred[1][idx]
        p2d = p2d.reshape(2,16)
        aa = p2d * 1
        p2d = (p2d - p2d.mean(-1)[:, None]).reshape(-1)
        p3d = self.data[1][idx]
        confidences = np.ones_like(self.pred[2][idx])
        annot_p2d = self.data[0][idx]
        annot_p2d = annot_p2d.reshape(2,16)

        annot_p2d = (annot_p2d - annot_p2d.mean(-1)[:, None]).reshape(-1)
        img_path = self.data[2][idx]
        sample = [p2d,confidences,p3d, annot_p2d, img_path, aa]

        return sample

class H36MDataset_temp(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, gt = True, normalize_2d=True, frame_size = 3):
        data_folder = './data/'
        if gt:
            fname = data_folder + 'h36m_p2d_gt.pkl'
        else:
            fname = data_folder + 'h36m_p2d_hr.pkl'

        pickle_off = open(fname, "rb")
        self.data_p2d = pickle.load(pickle_off)

        fname = data_folder + 'h36m_p3d.pkl'
        pickle_off = open(fname, "rb")
        self.data_p3d = pickle.load(pickle_off)
        self.data = get_temp_frame(frame_size, self.data_p2d, self.data_p3d)
        # index = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 14, 15, 16, 11, 12, 13]

        self.data[0] = np.transpose(self.data[0],(0,1,3,2))
        self.data[1] = np.transpose(self.data[1], (0, 2, 1))
        if normalize_2d:
            self.data[0] -= 512
            self.data[0] /= 512


        self.data[1] = self.data[1].reshape(-1,48)
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.data[0][idx]
        # p2d = self.data[0][idx]
        p3d = self.data[1][idx]
        confidences = np.ones(16)

        sample = [p2d,confidences,p3d]

        return sample

class H36MDataset_temp_pred(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname= data_folder + 'h36m_pred_test.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)
        self.data[0] = np.transpose(self.data[0], (0, 2, 1))
        self.data[1] = np.transpose(self.data[1], (0, 2, 1))

        if normalize_2d:
            self.data[0] -= 512
            self.data[0] /= 512


        self.data[1] = self.data[1].reshape(-1,48)
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx -= 1

        p2d_0 = self.data[0][idx + 1].reshape(32)
        p2d_1 = self.data[0][idx + 1].reshape(32)
        p2d_2 = self.data[0][idx + 1].reshape(32)

        p2d = np.concatenate([p2d_0, p2d_1, p2d_2], axis=0)
        # p2d = self.data[0][idx]
        p3d = self.data[1][idx + 1]
        confidences = np.ones(16)

        sample = [p2d, confidences, p3d]

        return sample


class PW3DDataset_temp(Dataset):
   

    def __init__(self, normalize_2d=True, frame_size = 3):
        data_folder = './data/'
        fname= data_folder + '3dpw_test.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)
        # index = [2, 1,0,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # fname = data_folder + 'data_3dpw_gt_test.npz'
        # self.dataset = np.load(fname)
        # self.data = []
        # self.data.append(np.array(self.dataset['positions_2d'])[:,index] * 512)
        # self.data.append(np.array(self.dataset['positions_3d'])[:, index] * 1000)
        self.frame_size = frame_size
        self.data[0] = np.transpose(self.data[0],(0,2,1))
        self.data[1] = np.transpose(self.data[1], (0, 2, 1))

        if normalize_2d:
            # self.data[0][:,0] -= 960
            # self.data[0][:, 1] -= 540
            self.data[0] /= 512
        self.data[1] = self.data[1].reshape(-1,48)
    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        try:
            p2d = []
            for i in range(self.frame_size):
                p2d.append(self.data[0][idx + i - int(self.frame_size/2)].reshape(32))

        except:
            p2d = []
            for i in range(self.frame_size):
                p2d.append(self.data[0][idx].reshape(32))

        p2d = np.stack(p2d, axis=0)
        p3d = self.data[1][idx]
        confidences = np.ones(16)

        sample = [p2d,confidences,p3d]

        return sample


class MPI_3DHP(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot2.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 128
            self.data[0] /= 128
        self.data[1] = self.data[1].reshape(-1,48)


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.pred[1][idx]
        p2d = p2d.reshape(2,16)
        aa = p2d * 1
        p2d = (p2d - p2d.mean(-1)[:, None]).reshape(-1)
        p3d = self.data[1][idx]
        confidences = np.ones_like(self.pred[2][idx])
        annot_p2d = self.data[0][idx]
        annot_p2d = annot_p2d.reshape(2,16)

        annot_p2d = (annot_p2d - annot_p2d.mean(-1)[:, None]).reshape(-1)
        img_path = self.data[2][idx]
        sample = [p2d,confidences,p3d, annot_p2d, img_path, aa]

        return sample

class MPI_3DHP_unbox(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full_withbox.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))
        box = np.array(self.data[3])
        self.data[0] *= (box[:, 1] - box[:, 0])[:, None, None] / 256
        self.data[0][:, :, 0] += box[:, 2][:, None]
        self.data[0][:, :, 1] += box[:, 0][:, None]


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 512
            self.data[0] /= 512
        self.data[1] = self.data[1].reshape(-1,48)


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.pred[1][idx]
        p2d = p2d.reshape(2,16)
        aa = p2d * 1
        p2d = (p2d - p2d.mean(-1)[:, None]).reshape(-1)
        p3d = self.data[1][idx]
        confidences = np.ones_like(self.pred[2][idx])
        annot_p2d = self.data[0][idx]
        annot_p2d = annot_p2d.reshape(2,16)

        img_path = self.data[2][idx]
        sample = [p2d,confidences,p3d, annot_p2d, img_path, aa]

        return sample

class MPI_3DHP_temp(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'mpi_test_temp.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)
        self.data[0] = np.array(self.data[0])
        self.data[1] = np.array(self.data[1])
        # if normalize_2d:
        #     self.data[0] -= 1024
        #     self.data[0] /= 512
        self.data[1] = self.data[1].reshape(-1, 48)

    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.data[0][idx]

        if self.data[2][idx] <= 3 :
            p2d -= 1024
            p2d /= 512
        else:
            p2d[:,0] -= 960
            p2d[:, 1] -= 540
            p2d /= 512

        p3d = self.data[1][idx]
        confidences = np.ones(16)

        sample = [p2d, confidences, p3d]

        return sample

class MPI_3DHP_img(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot2.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))
        self.root1 = '..'
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.target_type = 'gaussian'
        self.num_joints = 16
        self.heatmap_size = np.array([256, 256])
        self.image_size = np.array([256, 256])
        self.sigma = 2


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 128
            self.data[0] /= 128
        self.data[1] = self.data[1].reshape(-1,48)

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        '''
        joints_vis = np.ones_like(joints)
        target_weight = np.ones((16 , 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]



        return target


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.pred[1][idx]
        p2d = p2d.reshape(2,16)
        annot_p2d = self.data[0][idx]
        annot_p2d = annot_p2d.reshape(2,16)
        p2d = p2d - p2d.mean(1)[:,None] + annot_p2d.mean(1)[:,None]
        aa = p2d * 1
        aa *= 128
        aa += 128
        aa = np.transpose(aa,(1, 0))
        aa = np.stack([aa[:,0], aa[:,1], np.ones_like(aa[:,0])], axis=-1)
        confidences = self.pred[2][idx]
        heatmap = self.generate_target(aa) * confidences[:,None,None]

        p3d = self.data[1][idx]



        annot_p2d = (annot_p2d - annot_p2d[:, 6][:, None]).reshape(-1)
        im_path = os.path.join(self.root1, self.data[2][idx])
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)

        img = self.transform(img)
        sample = [p2d,confidences,p3d, annot_p2d, img, heatmap]

        return sample


class MPI_3DHP_img_annot(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot2.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))
        self.root1 = '..'
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.target_type = 'gaussian'
        self.num_joints = 16
        self.heatmap_size = np.array([256, 256])
        self.image_size = np.array([256, 256])
        self.sigma = 2


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 128
            self.data[0] /= 128
        self.data[1] = self.data[1].reshape(-1,48)

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        '''
        joints_vis = np.ones_like(joints)
        target_weight = np.ones((16 , 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]



        return target


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.data[0][idx]
        p2d = p2d.reshape(2,16)
        aa = p2d * 1
        aa *= 128
        aa += 128
        aa = np.transpose(aa,(1, 0))
        aa = np.stack([aa[:,0], aa[:,1], np.ones_like(aa[:,0])], axis=-1)
        heatmap = self.generate_target(aa)


        p3d = self.data[1][idx]
        confidences = np.ones_like(self.pred[2][idx])
        annot_p2d = self.data[0][idx]

        annot_p2d = annot_p2d.reshape(2,16)
        aff = np.array([[1, 0, -annot_p2d[0][6]*128],
                        [0, 1, -annot_p2d[1][6]*128 ]])

        annot_p2d = (annot_p2d - annot_p2d[:, 6][:, None]).reshape(-1)
        im_path = os.path.join(self.root1, self.data[2][idx])
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)
        # img = cv2.warpAffine(
        #     img,
        #     np.array(aff),
        #     (256, 256),
        #     flags=cv2.INTER_LINEAR)
        img = self.transform(img)
        sample = [p2d,confidences,p3d, annot_p2d, img, heatmap]

        return sample


class MPI_3DHP_img_morph(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, normalize_2d=True):
        data_folder = './data/'
        fname = data_folder + 'annont_test_full.pkl'
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_rot2.pkl'
        pickle_off = open(fname, "rb")
        self.pred = pickle.load(pickle_off)

        fname = data_folder + 'pred_test_morph.pkl'
        pickle_off = open(fname, "rb")
        self.morph = pickle.load(pickle_off)
        self.morph = self.morph.cpu()
        p2d_data = np.array(self.data[0])
        self.data[0] = np.array(self.data[0])
        self.pred[1] = np.transpose(np.array(self.pred[1]), (0,2,1)) / 8
        self.pred[2] = np.array(self.pred[2])
        p2d = self.pred[1] * 1
        score = self.pred[2]
        self.data[1] = np.transpose(np.array(self.data[1]), (0,2,1))
        self.root1 = '..'
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.target_type = 'gaussian'
        self.num_joints = 16
        self.heatmap_size = np.array([256, 256])
        self.image_size = np.array([256, 256])
        self.sigma = 2


        p3d = self.data[1] * 1
        index = [2,1,0,3,4,5,6,7,8,9,12,11,10,13,14,15]
        for i, j in enumerate(index):
            self.pred[1][:,:,i] = p2d[:,:, j]
            self.data[1][:, :, i] = p3d[:, :, j]
            self.pred[2][:,i] = score[:,j]
            self.data[0][:,i] = p2d_data[:,j]
        self.data[0] = np.transpose(np.array(self.data[0]), (0, 2, 1)).reshape(-1,32)

        if normalize_2d:
            self.pred[1] = (self.pred[1] - 128).reshape(-1,32)
            self.pred[1] /= 128
            self.data[0] -= 128
            self.data[0] /= 128
        self.data[1] = self.data[1].reshape(-1,48)

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        '''
        joints_vis = np.ones_like(joints)
        target_weight = np.ones((16 , 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]



        return target


    def __len__(self):
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p2d = self.morph[idx]
        p2d = p2d.reshape(2,16)
        annot_p2d = self.data[0][idx]
        annot_p2d = annot_p2d.reshape(2,16)
        p2d = p2d - p2d.mean(1)[:,None] + annot_p2d.mean(1)[:,None]
        print(p2d)
        print(annot_p2d)
        exit()
        aa = p2d * 1
        aa *= 128
        aa += 128
        aa = np.transpose(aa,(1, 0))
        aa = np.stack([aa[:,0], aa[:,1], np.ones_like(aa[:,0])], axis=-1)
        confidences = self.pred[2][idx]
        heatmap = self.generate_target(aa) * confidences[:,None,None]

        p3d = self.data[1][idx]



        annot_p2d = (annot_p2d - annot_p2d[:, 6][:, None]).reshape(-1)
        im_path = os.path.join(self.root1, self.data[2][idx])
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)

        img = self.transform(img)
        sample = [self.morph[idx],confidences,p3d, annot_p2d, img, heatmap]

        return sample