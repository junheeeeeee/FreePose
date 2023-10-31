import torch
from types import SimpleNamespace
from utils.utils import p3d_no_scale, freepose
from utils.evaluate import PCK_3d


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



def train_free(model, criterion, optimizer,args, losses, max_norm=True):
    ac_3d = AverageMeter()
    model.train()       

    p3d, p2d_temp, distance = freepose(args.batch_size * args.data_size, args.frame_size)
    p3d = p3d.reshape(args.batch_size, args.data_size, -1)
    p2d_temp = p2d_temp.reshape(args.batch_size, args.data_size, -1)
    distance = distance.reshape(args.batch_size, args.data_size, -1)

    for i in range(args.data_size):
        ramdom_p3d, ramdom_p2d_temp = p3d[:,i], p2d_temp[:,i]

        ## input_shape (batch_size, frame_size * 2 * joints_num)
        pred = model(ramdom_p2d_temp)

        losses.p3d = criterion( p3d_no_scale(pred[0].reshape(-1, 3, 16)) , p3d_no_scale(
            ramdom_p3d.reshape(-1, 3, 16) - ramdom_p3d.reshape(-1, 3, 16)[:, :, 6][:, :, None]))
        losses.p3d = ((losses.p3d.sum(1) + 1e-9) ** 0.5 * distance[:, i]).mean()
        losses.loss = losses.p3d
        optimizer.zero_grad()
        losses.loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        optimizer.step()

        with torch.no_grad():

            annot_p3d = (ramdom_p3d.reshape(-1, 3, 16)).permute(0, 2, 1).detach().cpu().numpy()

            pred_p3d = pred[0].reshape(-1, 3, 16).permute(0, 2, 1).detach().cpu().numpy()
            pred_p3d -= pred_p3d.mean(1)[:, None, :]
            annot_p3d -= annot_p3d.mean(1)[:, None, :]

            _, a3d = PCK_3d(pred_p3d, annot_p3d)
            ac_3d.update(a3d.mean(), ramdom_p3d.size(0))

    return ac_3d