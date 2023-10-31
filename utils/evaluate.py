import numpy as np

def accuracy_3d(pred, gt, alignment='scale'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
    Returns:
        tuple: A tuple containing joint position errors
        - mpjpe (float|np.ndarray[N]): mean per-joint position error.
        - p-mpjpe (float|np.ndarray[N]): mpjpe after rigid alignment with the
            ground truth
    """

    # size = pred.shape
    # pred = (pred[:, :, :3] - pred[:, 2, :3].reshape(size[0], 1, 3))
    # gt = (gt[:, :, :3] - gt[:, 2, :3].reshape(size[0], 1, 3))
    # p_len = ((pred[:, 0, :3] - pred[:, 1, :3]) ** 2).sum(1) ** 0.5
    # t_len = ((gt[:, 0, :3] - gt[:, 1, :3]) ** 2).sum(1) ** 0.5
    # p_len = p_len.reshape(-1, 1, 1)
    # t_len = t_len.reshape(-1, 1, 1)
    # len = t_len / p_len
    # pred = pred * len
    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1).mean()

    return error

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def PCK_3d(pred, gt, alignment='scale', threshold=150):
    """Calculate the Percentage of Correct Keypoints (3DPCK) w. or w/o rigid
    alignment.
    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV`2017
    More details can be found in the `paper
    <https://arxiv.org/pdf/1611.09813>`__.
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
        threshold:  If L2 distance between the prediction and the groundtruth
            is less then threshold, the predicted result is considered as
            correct. Default: 0.15 (m).
    Returns:
        pck: percentage of correct keypoints.
    """
    # size = pred.shape
    # pred = (pred[:, :, :3] - pred[:, 2, :3].reshape(size[0], 1, 3))
    # gt = (gt[:, :, :3] - gt[:, 2, :3].reshape(size[0], 1, 3))
    # p_len = ((pred[:, 0, :3] - pred[:, 1, :3]) ** 2).sum(1) ** 0.5
    # t_len = ((pred[:, 0, :3] - pred[:, 1, :3]) ** 2).sum(1) ** 0.5
    # p_len = p_len.reshape(-1, 1, 1)
    # t_len = t_len.reshape(-1, 1, 1)
    # len = t_len / p_len
    # pred = pred * len
    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)

        scale_factor = pred_dot_gt / pred_dot_pred
        # print(scale_factor)
        # exit()
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)
    pck = (error < threshold).astype(np.float32).mean() * 100

    return pck , error


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

class Metrics:
    def __init__(self, init=0):
        self.init = init

    def mpjpe(self, p_ref, p, scale=True, mean_align=True):
        import numpy as np

        # reshape pose if necessary
        if p.shape[0] == 1:
            p = p.reshape(3, int(p.shape[1] / 3))
        if p_ref.shape[0] == 1:
            p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

        if mean_align:
            p = p - p.mean(axis=1, keepdims=True)
            p_ref = p_ref - p_ref.mean(axis=1, keepdims=True)
        if scale:
            scale_p = np.linalg.norm(p.reshape(-1, 1), ord=2)
            scale_p_ref = np.linalg.norm(p_ref.reshape(-1, 1), ord=2)
            scale = scale_p_ref/scale_p
            p = p * scale

        sum_dist = 0

        for i in range(p.shape[1]):
            sum_dist += np.linalg.norm(p[:, i] - p_ref[:, i], 2)

        err = np.sum(sum_dist) / p.shape[1]

        return err

    def pmpjpe(self, p_ref, p):
        # reshape pose if necessary
        if p.shape[0] == 1:
            p = p.reshape(3, int(p.shape[1] / 3))

        if p_ref.shape[0] == 1:
            p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

        d, Z, tform = self.procrustes(p_ref.T, p.T)
        err = self.mpjpe(p_ref, Z.T)

        return err

    def PCK(self, p_ref, p):
       # reshape pose if necessary
       if p.shape[0] == 1:
           p = p.reshape(3, int(p.shape[1] / 3))

       if p_ref.shape[0] == 1:
           p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

       d, Z, tform = self.procrustes(p_ref.T, p.T)

       err = self.mpjpe(p_ref, Z.T)

       return err

    def procrustes(self, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.
        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.
            d, Z, [tform] = procrustes(X, Y)
        Inputs:
        ------------
        X, Y
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.
        scaling
            if False, the scaling component of the transformation is forced
            to 1
        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.
        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()
        Z
            the matrix of transformed Y-values
        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y
        """

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection != 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2

            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform

if __name__ == "__main__":
    M = Metrics()
    a = np.random.randn(3,16)
    b= np.random.randn(3,16)
    aa = M.mpjpe(a, b)
    print(aa)
    aa = M.pmpjpe(a, b)
    print(aa)
