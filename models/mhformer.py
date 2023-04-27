import torch
import torch.nn as nn
from einops import rearrange
from models.module.trans import Transformer as Transformer_encoder
from models.module.trans_hypothesis import Transformer as Transformer_hypothesis
from utils.kinematics import forward_kinematics_ver2

class Model(nn.Module):
    def __init__(self, frame):
        super().__init__()
        ## setting
        self.frames = frame
        self.n_joints = 16
        self.channel = 512
        self.out_joints = 16
        self.d_hid = 1024
        self.layers = 3

        ## MHG
        self.norm_1 = nn.LayerNorm(self.frames)
        self.norm_2 = nn.LayerNorm(self.frames)
        self.norm_3 = nn.LayerNorm(self.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)

        ## Embedding
        if self.frames > 27:
            self.embedding_1 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2 * self.out_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(self.layers, self.channel, self.d_hid, length=self.frames)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(self.channel * 3, momentum=0.1),
            nn.Conv1d(self.channel * 3, 3 * self.out_joints, kernel_size=1)
        )

    def forward(self, x):
        x = x.reshape(-1, self.frames, 2, self.n_joints)
        x = x.permute(0,1,3,2)
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        print("MHG")
        print('x_1 shape : ', x_1.shape)
        print('x_2 shape : ', x_2.shape)
        print('x_3 shape : ', x_3.shape)
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()
        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3)
        print("SHR & CHI")
        print('x shape : ', x.shape)
        ## Regression
        x = x.permute(0, 2, 1).contiguous() 
        x = self.regression(x)
        print('regression shape : ', x.shape)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x


class MH6d_6d(nn.Module):
    def __init__(self, frame):
        super().__init__()
        ## setting
        self.frames = frame
        self.n_joints = 16
        self.channel = 512
        self.out_joints = 15
        self.d_hid = 1024
        self.layers = 3

        self.embedding_6d = nn.Conv1d(2 * self.n_joints, 6 * self.out_joints + 9, kernel_size=1)
        ## MHG
        self.Transformer_encoder_1 = Transformer_encoder(4, self.frames, self.frames * 2, length=6 * self.out_joints + 9, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, self.frames, self.frames * 2, length=6 * self.out_joints + 9, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, self.frames, self.frames * 2, length=6 * self.out_joints + 9, h=9)

        ## Embedding
        if self.frames > 27:
            self.embedding_1 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(self.layers, self.channel, self.d_hid, length=self.frames)

        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(self.channel * 3, momentum=0.1),
            nn.Conv1d(self.channel * 3, 6 * self.out_joints + 9, kernel_size=1)
        )

    def forward(self, x):
        x = x.reshape(-1, self.frames, 2, self.n_joints)
        x = x.permute(0,1,3,2)
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()
        x = self.embedding_6d(x)

        ## MHG
        x_1 = x + self.Transformer_encoder_1(x)
        x_2 = x_1 + self.Transformer_encoder_2(x_1)
        x_3 = x_2 + self.Transformer_encoder_3(x_2)

        x_1 = x_1.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)
        x_2 = x_2.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)
        x_3 = x_3.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)

        x_1 = forward_kinematics_ver2(x_1[:, 9:], x_1[:,:9]).reshape(B,F,-1).permute(0, 2, 1)
        x_2 = forward_kinematics_ver2(x_2[:, 9:], x_2[:,:9]).reshape(B,F,-1).permute(0, 2, 1)
        x_3 = forward_kinematics_ver2(x_3[:, 9:], x_3[:,:9]).reshape(B,F,-1).permute(0, 2, 1)
        # print("MHG")
        # print('x_1 shape : ', x_1.shape)
        # print('x_2 shape : ', x_2.shape)
        # print('x_3 shape : ', x_3.shape)
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()
        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3)
        # print("SHR & CHI")
        # print('x shape : ', x.shape)
        ## Regression
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        # print('regression shape : ', x.shape)
        x = x.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)

        x = forward_kinematics_ver2(x[:, 9:], x[:,:9])
        x = x.reshape(B,F,-1)
        return [x[:, int(F/2)]]
    

class MH3d_6d(nn.Module):
    def __init__(self, frame):
        super().__init__()
        ## setting
        self.frames = frame
        self.n_joints = 16
        self.channel = 512
        self.out_joints = 15
        self.d_hid = 1024
        self.layers = 3

        ## MHG
        self.norm_1 = nn.LayerNorm(self.frames)
        self.norm_2 = nn.LayerNorm(self.frames)
        self.norm_3 = nn.LayerNorm(self.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, self.frames, self.frames * 2, length=3 * self.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, self.frames, self.frames * 2, length=3 * self.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, self.frames, self.frames * 2, length=3 * self.n_joints, h=9)

        ## Embedding
        if self.frames > 27:
            self.embedding_1 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(3 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(self.layers, self.channel, self.d_hid, length=self.frames)

        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(self.channel * 3, momentum=0.1),
            nn.Conv1d(self.channel * 3, 6 * self.out_joints + 9, kernel_size=1)
        )

    def forward(self, x):
        x = x.reshape(-1, self.frames, 2, self.n_joints)
        x = x.permute(0,1,3,2)
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        x_0 = torch.zeros_like(x)
        x = torch.concat([x, x_0[:,:16]], dim = 1)

        ## MHG
        x_1 = x + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1))
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        # print("MHG")
        # print('x_1 shape : ', x_1.shape)
        # print('x_2 shape : ', x_2.shape)
        # print('x_3 shape : ', x_3.shape)
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()
        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3)
        # print("SHR & CHI")
        # print('x shape : ', x.shape)
        ## Regression
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        # print('regression shape : ', x.shape)
        x = x.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)

        x = forward_kinematics_ver2(x[:, 9:], x[:,:9])
        x = x.reshape(B,F,-1)
        return [x[:, int(F/2)]]
    

class MH_6d(nn.Module):
    def __init__(self, frame):
        super().__init__()
        ## setting
        self.frames = frame
        self.n_joints = 16
        self.channel = 512
        self.out_joints = 15
        self.d_hid = 1024
        self.layers = 3

        ## MHG
        self.norm_1 = nn.LayerNorm(self.frames)
        self.norm_2 = nn.LayerNorm(self.frames)
        self.norm_3 = nn.LayerNorm(self.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, self.frames, self.frames * 2, length=2 * self.n_joints, h=9)

        ## Embedding
        if self.frames > 27:
            self.embedding_1 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2 * self.n_joints, self.channel, kernel_size=1),
                nn.BatchNorm1d(self.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(self.layers, self.channel, self.d_hid, length=self.frames)

        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(self.channel * 3, momentum=0.1),
            nn.Conv1d(self.channel * 3, 6 * self.out_joints + 9, kernel_size=1)
        )

    def forward(self, x):
        x = x.reshape(-1, self.frames, 2, self.n_joints)
        x = x.permute(0,1,3,2)
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG
        x_1 = x + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1))
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        # x_1 = forward_kinematics_ver2(x_1)
        # x_2 = forward_kinematics_ver2(x_2)
        # x_3 = forward_kinematics_ver2(x_3)
        # print("MHG")
        # print('x_1 shape : ', x_1.shape)
        # print('x_2 shape : ', x_2.shape)
        # print('x_3 shape : ', x_3.shape)
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()
        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3)
        # print("SHR & CHI")
        # print('x shape : ', x.shape)
        ## Regression
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        # print('regression shape : ', x.shape)
        x = x.permute(0,2,1).reshape(-1, 6 * self.out_joints + 9)

        x = forward_kinematics_ver2(x[:, 9:], x[:,:9])
        x = x.reshape(B,F,-1)
        return [x[:, int(F/2)]]



# test code here.
if __name__ == '__main__':
    frame = 9
    input = torch.randn((12, frame * 2 * 16), requires_grad=True).cuda()
    target = torch.randn((12, 16, 3), requires_grad=True).cuda()
    model = Model(frame).cuda()
    loss = nn.MSELoss()
    pre = model(input)
    print(pre.shape)
    # l = loss(pre, target)
    # l.backward()

    print('ss')




