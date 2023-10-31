from utils.utils import AverageMeter
import torch
from utils.evaluate import PCK_3d

def evaluation(data_loder, model, args):
    model.eval()
    mpjpe = AverageMeter()
    p_mpjpe = AverageMeter()
    pck = AverageMeter() 

    with torch.no_grad():
        for i, sample in enumerate(data_loder):

            inp_poses = sample[0].cuda().type(torch.float32)
            ## input_shape (batch_size, frame_size * 2 * joints_num)
            inp_poses = inp_poses.reshape(-1, args.frame_size * 32)

            pred = model(inp_poses)

            pred_poses = torch.transpose(pred[0].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            pred_poses = pred_poses - pred_poses.mean(1)[:, None, :]

            target = torch.transpose(sample[2].view(-1, 3, 16), 1, 2).detach().cpu().numpy()
            target = target - target.mean(1)[:, None, :]

            Ppck, pmpj = PCK_3d(pred_poses, target, alignment='procrustes')
            p_mpjpe.update(pmpj.mean(), inp_poses.size(0))
            pckk, mpj = PCK_3d(pred_poses, target)

            mpjpe.update(mpj.mean(), inp_poses.size(0))
            pck.update(Ppck, inp_poses.size(0))
    
    return mpjpe, p_mpjpe, pck