import torch

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def forward_kinematics(a , b):
    batch = a.shape[0]
    b = torch.nn.Softmax(-1)(b) 
    pose = torch.zeros((batch, 3, 16)).cuda()
    structure_index = [[0, 3], [1, 4], [2, 5], [7], [8], [9], [10, 13], [11, 14], [12, 15]]
    connections = [[6, 0], [6, 3], [6, 7], [0, 1], [1, 2], [3, 4], [4, 5],  [7, 10], [10, 11], [11, 12], [7, 13],
                   [13, 14], [14, 15], [7, 8], [8, 9]]
    nn = 0


    for n, i in enumerate(structure_index):
        for j in i:
            pose[:, :, j] += 1
            pose[:, :, j] *= b[:,[n]]
            pose[:, :, j] = (compute_rotation_matrix_from_ortho6d(a[:,nn]) @ pose[:, :, [j]]).squeeze()
            nn += 1
    for index in connections:
        pose[:,:,index[1]] += pose[:,:,index[0]]

    return pose

def forward_kinematics_ver2(a , b):
    ## input shape : batch, 15 * 6
    b = torch.nn.Softmax(-1)(b)
    batch = a.shape[0]
    a = a.reshape(batch, 15, 6)
    index = ['right', 'down', 'down', 'left', 'down', 'down', 'none', 'up', 'up', 'up', 'right', 'right', 'right', 'left', 'left', 'left']
    pose = torch.zeros((batch, 3, 16)).cuda()
    for i in [0, 3, 7]:
        if index[i] == 'right':
            pose[:, 0, i] = 1
        elif index[i] == 'left':
            pose[:, 0, i] = -1
        elif index[i] == 'down':
            pose[:, 1, i] = -1
        elif index[i] == 'up':
            pose[:, 1, i] = 1
        else:
            pass

    bone_index = [[0, 3], [1, 4], [2, 5], [7], [8], [9], [10, 13], [11, 14], [12, 15]]
    connections = [[0, 0], [3, 3], [7, 7], [0, 1], [1, 2], [3, 4], [4, 5], [7, 10], [10, 11], [11, 12], [7, 13],
                   [13, 14], [14, 15], [7, 8], [8, 9]]

    for n, index in enumerate(connections):
        if index[1] == 3:
            pose[:, :, index[1]] = pose[:, :, 0] * -1
        else:
            pose[:,:,index[1]] = (compute_rotation_matrix_from_ortho6d(a[:, n]) @ pose[:,:,[index[0]]]).squeeze() * 1
    for n, i in enumerate(bone_index):
        for j in i:
            pose[:, :, j] *= b[:,[n]]


    for index in connections:
        pose[:,:,index[1]] += pose[:,:,index[0]]

    return pose
