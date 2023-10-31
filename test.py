
import torch.optim

from torch.utils import data

from utils.data_val import H36MDataset_temp
from utils.data_val import PW3DDataset_temp as ValDataset3
from utils.data_val import SkyDataset_temp as ValDataset4

from types import SimpleNamespace

from utils.evaluate import PCK_3d, accuracy_3d, p_mpjpe , Metrics

# from utils.get_3d_skelton import get_3d_skeleton_for_canon as get_3d



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

data_folder = './data/'
config.datafile = data_folder + 'pred_train.pkl'

config.valdatafile = data_folder + 'annot_test_full.pkl'

# loading the H36M dataset


val_dataset2 = H36MDataset_temp(normalize_2d=True)
val_dataset3 = ValDataset3(normalize_2d=True)



val_loader2 = data.DataLoader(val_dataset2, batch_size=2048, shuffle=False, num_workers=6)
val_loader3 = data.DataLoader(val_dataset3, batch_size=2048, shuffle=False, num_workers=6)


cc = 0
ac_2d = AverageMeter()
ac_3d = AverageMeter()

H36M_mpj3d = AverageMeter()

H36M_pck_3d = AverageMeter()
H36M_Ppck_3d = AverageMeter()
multi_3d = AverageMeter()

H36M_P_mpj3d = AverageMeter()

# cv2.namedWindow('origin', cv2.WINDOW_AUTOSIZE)


# scores = np.concatenate(scores, axis=0)
# x = np.arange(0,2869)
#
# plt.plot(x,scores,'r')
# plt.show()
# exit()
model = torch.load('output/sota_model_H36M_59.8.pt').cuda()
model.eval()

with torch.no_grad():
    for i, sample in enumerate(val_loader2):
        # not the most elegant way to extract the dictionary


        inp_poses = sample[0].cuda().type(torch.float32)


        inp_poses = inp_poses.reshape(-1,3*32)

        # predict 3d poses

        pred = model(inp_poses)
        # pred_poses = get_3d_point(pred[0])
        pred_poses = torch.transpose(pred[0].view(-1, 3, 16), 1, 2).detach().cpu().numpy()


        target = torch.transpose(sample[2].view(-1, 3, 16), 1, 2).detach().cpu().numpy()



        pred_poses = pred_poses - pred_poses.mean(1)[:, None, :]
        target = target - target.mean(1)[:, None, :]
        Ppck,pmpj = PCK_3d(pred_poses, target, alignment = 'procrustes')
        H36M_P_mpj3d.update(pmpj.mean(), inp_poses.size(0))

        pckk, mpj = PCK_3d(pred_poses, target)
        # print(mpj.mean(0))
        H36M_mpj3d.update(mpj.mean(), inp_poses.size(0))
        H36M_pck_3d.update(pckk, inp_poses.size(0))



msg = 'H36M :\t' \
          'n-mpjpe {acc.avg:.1f}mm\t' \
          'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)
print(msg)
H36M_mpj3d.reset()
H36M_P_mpj3d.reset()
H36M_pck_3d.reset()

model = torch.load('output/sota_model_3DPW_57.0.pt').cuda()
model.eval()

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
        # print(mpj.mean(0))
        H36M_mpj3d.update(mpj.mean(), inp_poses.size(0))
        H36M_pck_3d.update(pckk, inp_poses.size(0))


msg = '3DPW :\t' \
      'n-mpjpe {acc.avg:.1f}mm\t' \
      'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)

print(msg)

H36M_mpj3d.reset()
H36M_P_mpj3d.reset()
H36M_pck_3d.reset()



# if not (epoch+1) % 30:
#     scheduler.step()
# scheduler.step(H36M_mpj3d.avg)


