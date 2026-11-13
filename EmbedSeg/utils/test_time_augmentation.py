import torch
import numpy as np


def to_cuda(im_numpy):
    """
    Converts 2D Numpy Image on CPU to PyTorch Tensor on GPU, with an extra dimension

    Parameters
    -------

    im_numpy: numpy array (YX)


    Returns
    -------
        Pytorch Tensor (1YX)

    """
    im_numpy = im_numpy[np.newaxis, ...]
    return torch.from_numpy(im_numpy).float().cuda()


def to_numpy(im_cuda):
    """
    Converts PyTorch Tensor on GPU to Numpy Array on CPU

    Parameters
    -------

    im_cuda: PyTorch tensor


    Returns
    -------
        numpy array

    """
    return im_cuda.cpu().detach().numpy()


def process_flips(im_numpy):
    """
    Converts the model output (5YX) so that y-offset is correctly handled
    (x-offset, y-offset, x-margin bandwidth, y-margin bandwidth, seediness score)

    Parameters
    -------

    im_numpy: Numpy Array (5YX)


    Returns
    -------
    im_numpy_correct: Numpy Array (5YX)

    """
    im_numpy_correct = im_numpy
    im_numpy_correct[0, 1, ...] = (
        -1 * im_numpy[0, 1, ...]
    )  # because flipping is always along y-axis, so only the y-offset gets affected
    return im_numpy_correct


def to_cuda_3d(im_numpy):
    """
    Converts 3D Numpy Image on CPU to PyTorch Tensor on GPU, with an extra dimension

    Parameters
    -------

    im_numpy: numpy array (ZYX)


    Returns
    -------
        Pytorch Tensor (CZYX)

    """
    im_numpy = im_numpy[np.newaxis, ...]
    return torch.from_numpy(im_numpy).float().cuda()


def _apply_tta_2d_single(im, model):
    """
    Apply Test Time Augmentation for 2D Images

    Parameters
    -------

    im: Numpy Array (1CYX)
    model: PyTorch Model

    Returns
    -------
    PyTorch Tensor on GPU (15YX)

    """
    im_numpy = im.cpu().detach().numpy()  # BCYX
    im0 = im_numpy[0, ...]  # remove batch dimension, now CYX
    im1 = np.rot90(im0, 1, (1, 2))
    im2 = np.rot90(im0, 2, (1, 2))
    im3 = np.rot90(im0, 3, (1, 2))
    im4 = np.flip(im0, 1)
    im5 = np.flip(im1, 1)
    im6 = np.flip(im2, 1)
    im7 = np.flip(im3, 1)

    im0_cuda = to_cuda(im0)  # BCYX
    im1_cuda = to_cuda(np.ascontiguousarray(im1))
    im2_cuda = to_cuda(np.ascontiguousarray(im2))
    im3_cuda = to_cuda(np.ascontiguousarray(im3))
    im4_cuda = to_cuda(np.ascontiguousarray(im4))
    im5_cuda = to_cuda(np.ascontiguousarray(im5))
    im6_cuda = to_cuda(np.ascontiguousarray(im6))
    im7_cuda = to_cuda(np.ascontiguousarray(im7))

    output0 = model(im0_cuda)
    output1 = model(im1_cuda)
    output2 = model(im2_cuda)
    output3 = model(im3_cuda)
    output4 = model(im4_cuda)
    output5 = model(im5_cuda)
    output6 = model(im6_cuda)
    output7 = model(im7_cuda)

    # de-transform outputs
    output0_numpy = to_numpy(output0)
    output1_numpy = to_numpy(output1)
    output2_numpy = to_numpy(output2)
    output3_numpy = to_numpy(output3)
    output4_numpy = to_numpy(output4)
    output5_numpy = to_numpy(output5)
    output6_numpy = to_numpy(output6)
    output7_numpy = to_numpy(output7)

    # invert rotations and flipping

    output1_numpy = np.rot90(output1_numpy, 1, (3, 2))
    output2_numpy = np.rot90(output2_numpy, 2, (3, 2))
    output3_numpy = np.rot90(output3_numpy, 3, (3, 2))
    output4_numpy = np.flip(output4_numpy, 2)
    output5_numpy = np.flip(output5_numpy, 2)
    output5_numpy = np.rot90(output5_numpy, 1, (3, 2))
    output6_numpy = np.flip(output6_numpy, 2)
    output6_numpy = np.rot90(output6_numpy, 2, (3, 2))
    output7_numpy = np.flip(output7_numpy, 2)
    output7_numpy = np.rot90(output7_numpy, 3, (3, 2))

    # have to also process the offsets and covariance sensibly
    output0_numpy_correct = output0_numpy

    # note rotations are always [cos(theta) sin(theta); -sin(theta) cos(theta)]
    output1_numpy_correct = np.zeros_like(output1_numpy)
    output1_numpy_correct[0, 0, ...] = -output1_numpy[0, 1, ...]
    output1_numpy_correct[0, 1, ...] = output1_numpy[0, 0, ...]
    output1_numpy_correct[0, 2, ...] = output1_numpy[0, 3, ...]
    output1_numpy_correct[0, 3, ...] = output1_numpy[0, 2, ...]
    output1_numpy_correct[0, 4, ...] = output1_numpy[0, 4, ...]

    output2_numpy_correct = np.zeros_like(output2_numpy)
    output2_numpy_correct[0, 0, ...] = -output2_numpy[0, 0, ...]
    output2_numpy_correct[0, 1, ...] = -output2_numpy[0, 1, ...]
    output2_numpy_correct[0, 2, ...] = output2_numpy[0, 2, ...]
    output2_numpy_correct[0, 3, ...] = output2_numpy[0, 3, ...]
    output2_numpy_correct[0, 4, ...] = output2_numpy[0, 4, ...]

    output3_numpy_correct = np.zeros_like(output3_numpy)
    output3_numpy_correct[0, 0, ...] = output3_numpy[0, 1, ...]
    output3_numpy_correct[0, 1, ...] = -output3_numpy[0, 0, ...]
    output3_numpy_correct[0, 2, ...] = output3_numpy[0, 3, ...]
    output3_numpy_correct[0, 3, ...] = output3_numpy[0, 2, ...]
    output3_numpy_correct[0, 4, ...] = output3_numpy[0, 4, ...]

    output4_numpy_correct = process_flips(output4_numpy)

    output5_numpy_flipped = process_flips(output5_numpy)
    output5_numpy_correct = np.zeros_like(output5_numpy_flipped)
    output5_numpy_correct[0, 0, ...] = -output5_numpy_flipped[0, 1, ...]
    output5_numpy_correct[0, 1, ...] = output5_numpy_flipped[0, 0, ...]
    output5_numpy_correct[0, 2, ...] = output5_numpy_flipped[0, 3, ...]
    output5_numpy_correct[0, 3, ...] = output5_numpy_flipped[0, 2, ...]
    output5_numpy_correct[0, 4, ...] = output5_numpy_flipped[0, 4, ...]

    output6_numpy_flipped = process_flips(output6_numpy)
    output6_numpy_correct = np.zeros_like(output6_numpy_flipped)
    output6_numpy_correct[0, 0, ...] = -output6_numpy_flipped[0, 0, ...]
    output6_numpy_correct[0, 1, ...] = -output6_numpy_flipped[0, 1, ...]
    output6_numpy_correct[0, 2, ...] = output6_numpy_flipped[0, 2, ...]
    output6_numpy_correct[0, 3, ...] = output6_numpy_flipped[0, 3, ...]
    output6_numpy_correct[0, 4, ...] = output6_numpy_flipped[0, 4, ...]

    output7_numpy_flipped = process_flips(output7_numpy)
    output7_numpy_correct = np.zeros_like(output7_numpy_flipped)
    output7_numpy_correct[0, 0, ...] = output7_numpy_flipped[0, 1, ...]
    output7_numpy_correct[0, 1, ...] = -output7_numpy_flipped[0, 0, ...]
    output7_numpy_correct[0, 2, ...] = output7_numpy_flipped[0, 3, ...]
    output7_numpy_correct[0, 3, ...] = output7_numpy_flipped[0, 2, ...]
    output7_numpy_correct[0, 4, ...] = output7_numpy_flipped[0, 4, ...]

    output = np.concatenate(
        (
            output0_numpy_correct,
            output1_numpy_correct,
            output2_numpy_correct,
            output3_numpy_correct,
            output4_numpy_correct,
            output5_numpy_correct,
            output6_numpy_correct,
            output7_numpy_correct,
        ),
        0,
    )
    output = np.mean(output, 0, keepdims=True)  # 1 5 Y X
    return torch.from_numpy(output).float().cuda()


def _apply_tta_3d_single(im, model, index):
    """
    Apply Test Time Augmentation for 3D Images

    Parameters
    -------

    im: Numpy Array (1CYX)
    model: PyTorch Model
    index: int (0 to 15)

    Returns
    -------
    Numpy Array on CPU (17ZYX)

    """
    im_numpy = im.cpu().detach().numpy()  # BCZYX
    im_transformed = im_numpy[0, ...]  # remove batch dimension, now CZYX
    if index == 0:
        temp = np.rot90(im_transformed, 0, (2, 3))
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)  # BCZYX
        output_numpy = np.rot90(output_numpy, 0, (4, 3))
        output_numpy_correct = output_numpy
    elif index == 1:
        temp = np.rot90(im_transformed, 1, (2, 3))
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.rot90(output_numpy, 1, (4, 3))

        output_numpy_correct = np.zeros_like(output_numpy)  # BCZYX
        output_numpy_correct[0, 0, ...] = -output_numpy[0, 1, ...]
        output_numpy_correct[0, 1, ...] = output_numpy[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy[0, 6, ...]

    elif index == 2:
        temp = np.rot90(im_transformed, 2, (2, 3))
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.rot90(output_numpy, 2, (4, 3))

        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy[0, 0, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy[0, 1, ...]
        output_numpy_correct[0, 2, ...] = output_numpy[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy[0, 3, ...]
        output_numpy_correct[0, 4, ...] = output_numpy[0, 4, ...]
        output_numpy_correct[0, 5, ...] = output_numpy[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy[0, 6, ...]
    elif index == 3:
        temp = np.rot90(im_transformed, 3, (2, 3))
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.rot90(output_numpy, 3, (4, 3))

        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = output_numpy[0, 1, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy[0, 6, ...]
    elif index == 4:
        temp = np.rot90(im_transformed, 0, (2, 3))
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.rot90(output_numpy, 0, (4, 3))

        output_numpy_correct = output_numpy
        output_numpy_correct[0, 1, ...] = -1 * output_numpy[0, 1, ...]
    elif index == 5:
        temp = np.rot90(im_transformed, 1, (2, 3))  # CZYX
        temp = np.flip(temp, 2)  # 2 --> Y
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.rot90(output_numpy, 1, (4, 3))

        # fix flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]

        # fix rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 6:
        temp = np.rot90(im_transformed, 2, (2, 3))
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.rot90(output_numpy, 2, (4, 3))

        # fix flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]

        # fix rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 7:
        temp = np.rot90(im_transformed, 3, (2, 3))
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.rot90(output_numpy, 3, (4, 3))

        # fix flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]

        # fix rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 8:
        temp = np.rot90(im_transformed, 0, (2, 3))
        temp = np.flip(temp, 1)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 0, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy[0, 2, ...]
        output_numpy_correct = output_numpy_flipped

    elif index == 9:
        temp = np.rot90(im_transformed, 1, (2, 3))
        temp = np.flip(temp, 1)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 1, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 10:
        temp = np.rot90(im_transformed, 2, (2, 3))
        temp = np.flip(temp, 1)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 2, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 11:
        temp = np.rot90(im_transformed, 3, (2, 3))
        temp = np.flip(temp, 1)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 3, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 12:
        temp = np.rot90(im_transformed, 0, (2, 3))
        temp = np.flip(temp, 1)
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 0, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy_flipped[0, 2, ...]
        output_numpy_correct = output_numpy_flipped

    elif index == 13:
        temp = np.rot90(im_transformed, 1, (2, 3))
        temp = np.flip(temp, 1)  # CZYX
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 1, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy_flipped[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 14:
        temp = np.rot90(im_transformed, 2, (2, 3))
        temp = np.flip(temp, 1)
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 2, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy_flipped[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    elif index == 15:
        temp = np.rot90(im_transformed, 3, (2, 3))
        temp = np.flip(temp, 1)
        temp = np.flip(temp, 2)
        im_cuda = to_cuda_3d(np.ascontiguousarray(temp))
        output = model(im_cuda)
        output_numpy = to_numpy(output)
        output_numpy = np.flip(output_numpy, 3)
        output_numpy = np.flip(output_numpy, 2)
        output_numpy = np.rot90(output_numpy, 3, (4, 3))

        # process flip
        output_numpy_flipped = output_numpy
        output_numpy_flipped[0, 1, ...] = -1 * output_numpy[0, 1, ...]
        output_numpy_flipped[0, 2, ...] = -1 * output_numpy_flipped[0, 2, ...]

        # process rotation
        output_numpy_correct = np.zeros_like(output_numpy)
        output_numpy_correct[0, 0, ...] = output_numpy_flipped[0, 1, ...]
        output_numpy_correct[0, 1, ...] = -output_numpy_flipped[0, 0, ...]
        output_numpy_correct[0, 2, ...] = output_numpy_flipped[0, 2, ...]
        output_numpy_correct[0, 3, ...] = output_numpy_flipped[0, 4, ...]
        output_numpy_correct[0, 4, ...] = output_numpy_flipped[0, 3, ...]
        output_numpy_correct[0, 5, ...] = output_numpy_flipped[0, 5, ...]
        output_numpy_correct[0, 6, ...] = output_numpy_flipped[0, 6, ...]

    return output_numpy_correct


class _BlockReplayModel:
    """Wraps a model for blockwise test-time augmentation.

    The corrections of the TTA code (sign / permutation of offset and sigma
    channels) are written for one segmentation block of 5 (2D) or 7 (3D) channels.
    A tracking model has additional blocks of the same layout (offsets, sigma, one
    scalar), which need exactly the same correction. In the first pass the real
    model runs (and its outputs are recorded); in later passes the recorded
    outputs are replayed, restricted to one channel block.
    """

    def __init__(self, model, block_size):
        self.model = model
        self.block_size = block_size
        self.recorded = []
        self.replay = False
        self.block = 0
        self.k = 0

    def __call__(self, x):
        if not self.replay:
            out = self.model(x)
            self.recorded.append(out)
            return out
        out = self.recorded[self.k]
        self.k += 1
        start = self.block * self.block_size
        return out[:, start : start + self.block_size]

    @property
    def n_blocks(self):
        channels = self.recorded[0].shape[1]
        if channels > self.block_size and channels % self.block_size == 0:
            return channels // self.block_size
        return 1  # plain segmentation model: behave exactly as before


def _apply_blockwise(single, im, model, block_size, concatenate, *args):
    proxy = _BlockReplayModel(model, block_size)
    first = single(im, proxy, *args)
    if proxy.n_blocks == 1:
        return first
    parts = [first[:, :block_size]]
    proxy.replay = True
    for block in range(1, proxy.n_blocks):
        proxy.block, proxy.k = block, 0
        parts.append(single(im, proxy, *args))
    return concatenate(parts)


def apply_tta_2d(im, model):
    """Test Time Augmentation for 2D images (see `_apply_tta_2d_single`).
    Models with several 5-channel blocks (segmentation + tracking) are
    handled blockwise; a plain 5-channel model behaves as before."""
    return _apply_blockwise(
        _apply_tta_2d_single, im, model, 5, lambda parts: torch.cat(parts, 1)
    )


def apply_tta_3d(im, model, index):
    """Test Time Augmentation for 3D images (see `_apply_tta_3d_single`).
    Models with several 7-channel blocks (segmentation + tracking) are
    handled blockwise; a plain 7-channel model behaves as before."""
    return _apply_blockwise(
        _apply_tta_3d_single, im, model, 7, lambda parts: np.concatenate(parts, 1), index
    )
