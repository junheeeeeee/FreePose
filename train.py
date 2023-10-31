import torch
import torch.nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils import data
from utils.data_val import PW3DDataset_temp
from utils.data_val import H36MDataset_temp
import torch.optim as optim
from models.model_confidences import Lifter_non_3d as Lifter
from types import SimpleNamespace
from utils.config import get_parse_args
from utils.model_train import train_free
from utils.model_eval import evaluation
from models.mhformer import MH_6d, MH3d_6d, MH6d_6d
import os
from time import localtime, time, strftime

from torch.utils.tensorboard import SummaryWriter

def main(args):
    losses = SimpleNamespace()
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    if args.eval_h36m:
        h36m_dataset = H36MDataset_temp(normalize_2d=True, frame_size= args.frame_size)
        h36m_loader = data.DataLoader(h36m_dataset, batch_size=2048, shuffle=False, num_workers=args.num_workers)
        best_H = 300

    if args.eval_h36mp:
        h36mp_dataset = H36MDataset_temp(gt = False, normalize_2d=True, frame_size= args.frame_size)
        h36mp_loader = data.DataLoader(h36mp_dataset, batch_size=2048, shuffle=False, num_workers= args.num_workers)
        best_Hp = 300

    if args.eval_hp3d:
        pass
        best_3dhp = 300

    if args.eval_pw3d:
        pw3d_dataset = PW3DDataset_temp(normalize_2d = True, frame_size= args.frame_size)
        pw3d_loader = data.DataLoader(pw3d_dataset, batch_size=2048, shuffle=False, num_workers= args.num_workers)
        best_3dpw = 300
    

    print("==> Creating PoseNet model...")
    if args.posenet_name == 'simple':
        model = Lifter(args.frame_size).cuda()
    elif args.posenet_name == "mhformer":
        model = MH_6d(args.frame_size).cuda()
    elif args.posenet_name == "mh3d":
        model = MH3d_6d(args.frame_size).cuda()
    elif args.posenet_name == "mh6d":
        model = MH6d_6d(args.frame_size).cuda()
    else:
        print("nono")
        exit()
    print("==> Prepare optimizer...")
    criterion = torch.nn.MSELoss(reduction='none')
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    tm = localtime(time())
    writer = SummaryWriter(f"logs/{tm.tm_year}-{tm.tm_mon.zfill(2)}-{tm.tm_mday.zfill(2)}_{tm.tm_hour.zfill(2)}-{tm.tm_min.zfill(2)}-{tm.tm_sec.zfill(2)}\
_FreePose_Posenet={args.posenet_name}_frame_size={args.frame_size}_lr={args.lr}_batch_size={args.batch_size}")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        train_free(model, criterion, optimizer, args, losses)


        if args.eval_h36m:
            H36M_mpj3d, H36M_P_mpj3d, H36M_pck_3d = evaluation(h36m_loader, model, args)

            msg = 'H36M :\t' \
                'n-mpjpe {acc.avg:.1f}mm\t' \
                'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)
            print(msg)
            writer.add_scalar("Eval/H36M_nmpj3d", H36M_mpj3d.avg, epoch)
            writer.add_scalar("Eval/H36M_pmpj3d", H36M_P_mpj3d.avg, epoch)
            if best_H > H36M_mpj3d.avg:
                if os.path.isfile(f'output/model_lifter_single_H36M_{best_H}.pt'):
                    os.remove(f'output/model_lifter_single_H36M_{best_H}.pt')
                best_H = H36M_mpj3d.avg
                torch.save(model, f'output/model_lifter_single_H36M_{best_H}.pt')
                print('save_best')
                
        if args.eval_h36mp:
            H36M_mpj3d, H36M_P_mpj3d, H36M_pck_3d = evaluation(h36mp_loader, model, args)

            msg = 'H36Mp :\t' \
                'n-mpjpe {acc.avg:.1f}mm\t' \
                'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)
            print(msg)
            writer.add_scalar("Eval/H36Mp_nmpj3d", H36M_mpj3d.avg, epoch)
            writer.add_scalar("Eval/H36Mp_pmpj3d", H36M_P_mpj3d.avg, epoch)
            if best_Hp > H36M_mpj3d.avg:
                if os.path.isfile(f'output/model_lifter_single_H36Mpred_{best_Hp}.pt'):
                    os.remove(f'output/model_lifter_single_H36Mpred_{best_Hp}.pt')
                best_Hp = H36M_mpj3d.avg
                torch.save(model, f'output/model_lifter_single_H36Mpred_{best_Hp}.pt')
                print('save_bs')

        if args.eval_pw3d:
            H36M_mpj3d, H36M_P_mpj3d, H36M_pck_3d = evaluation(pw3d_loader, model, args)
        

            msg = '3DPW :\t' \
                'n-mpjpe {acc.avg:.1f}mm\t' \
                'P_mpjpe {pck.avg:.1f}mm'.format(acc=H36M_mpj3d, pck=H36M_P_mpj3d)

            print(msg)
            writer.add_scalar("Eval/3DPW_nmpj3d", H36M_mpj3d.avg, epoch)
            writer.add_scalar("Eval/3DPW_pmpj3d", H36M_P_mpj3d.avg, epoch)

            if best_3dpw > H36M_mpj3d.avg:
                if os.path.isfile(f'output/model_lifter_single_3DPW_{best_3dpw}.pt'):
                    os.remove(f'output/model_lifter_single_3DPW_{best_3dpw}.pt')
                best_3dpw = H36M_mpj3d.avg
                torch.save(model, f'output/model_lifter_single_3DPW_{best_3dpw}.pt')
                print('save_best')
                cc = 0

        print()
        print(f'Best = H : {best_H} Hp : {best_Hp} 3dpw : {best_3dpw}' )
        print()
    print('done')


if __name__ == '__main__':
    args = get_parse_args()
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    main(args)