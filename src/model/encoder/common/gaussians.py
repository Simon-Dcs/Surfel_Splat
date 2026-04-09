import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
import sys
from datetime import datetime
import numpy as np
import random

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

# def rotation_matrix_to_quaternion(R):
#     """
#     Convert 3x3 rotation matrix to quaternion [r, x, y, z]
#     Args:
#         R: rotation matrix [B, 3, 3]
#     Returns:
#         q: quaternion [B, 4]
#     """
#     batch_size = R.shape[0]
    
#     # Compute trace of rotation matrix
#     trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
#     q = torch.zeros((batch_size, 4), device=R.device)
    
#     mask_0 = trace > 0
#     mask_1 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & (~mask_0)
#     mask_2 = (R[:, 1, 1] > R[:, 2, 2]) & (~mask_0) & (~mask_1)
#     mask_3 = (~mask_0) & (~mask_1) & (~mask_2)
    
#     # Case 1: trace > 0
#     S_0 = torch.sqrt(trace + 1.0) * 2
#     q[mask_0, 0] = 0.25 * S_0
#     q[mask_0, 1] = (R[mask_0, 2, 1] - R[mask_0, 1, 2]) / S_0
#     q[mask_0, 2] = (R[mask_0, 0, 2] - R[mask_0, 2, 0]) / S_0
#     q[mask_0, 3] = (R[mask_0, 1, 0] - R[mask_0, 0, 1]) / S_0
    
#     # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
#     S_1 = torch.sqrt(1.0 + R[mask_1, 0, 0] - R[mask_1, 1, 1] - R[mask_1, 2, 2]) * 2
#     q[mask_1, 0] = (R[mask_1, 2, 1] - R[mask_1, 1, 2]) / S_1
#     q[mask_1, 1] = 0.25 * S_1
#     q[mask_1, 2] = (R[mask_1, 0, 1] + R[mask_1, 1, 0]) / S_1
#     q[mask_1, 3] = (R[mask_1, 0, 2] + R[mask_1, 2, 0]) / S_1
    
#     # Case 3: R[1,1] > R[2,2]
#     S_2 = torch.sqrt(1.0 + R[mask_2, 1, 1] - R[mask_2, 0, 0] - R[mask_2, 2, 2]) * 2
#     q[mask_2, 0] = (R[mask_2, 0, 2] - R[mask_2, 2, 0]) / S_2
#     q[mask_2, 1] = (R[mask_2, 0, 1] + R[mask_2, 1, 0]) / S_2
#     q[mask_2, 2] = 0.25 * S_2
#     q[mask_2, 3] = (R[mask_2, 1, 2] + R[mask_2, 2, 1]) / S_2
    
#     # Case 4: R[2,2] is largest diagonal
#     S_3 = torch.sqrt(1.0 + R[mask_3, 2, 2] - R[mask_3, 0, 0] - R[mask_3, 1, 1]) * 2
#     q[mask_3, 0] = (R[mask_3, 1, 0] - R[mask_3, 0, 1]) / S_3
#     q[mask_3, 1] = (R[mask_3, 0, 2] + R[mask_3, 2, 0]) / S_3
#     q[mask_3, 2] = (R[mask_3, 1, 2] + R[mask_3, 2, 1]) / S_3
#     q[mask_3, 3] = 0.25 * S_3
    
#     return q

def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion [r, x, y, z].
    Args:
        R: rotation matrix [B, 3, 3]
    Returns:
        q: quaternion [B, 4]
    """
    # Get batch dimension
    batch_size = R.shape[0]

    # Compute trace of rotation matrix
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros((*R.shape[:-2], 4), device=R.device)

    # Create masks
    mask_0 = trace > 0
    mask_1 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & (~mask_0)
    mask_2 = (R[..., 1, 1] > R[..., 2, 2]) & (~mask_0) & (~mask_1)
    mask_3 = (~mask_0) & (~mask_1) & (~mask_2)
    
    # Case 1: trace > 0
    S_0 = torch.sqrt(trace[mask_0] + 1.0) * 2
    q[mask_0, 0] = 0.25 * S_0
    q[mask_0, 1] = (R[mask_0, 2, 1] - R[mask_0, 1, 2]) / S_0
    q[mask_0, 2] = (R[mask_0, 0, 2] - R[mask_0, 2, 0]) / S_0
    q[mask_0, 3] = (R[mask_0, 1, 0] - R[mask_0, 0, 1]) / S_0
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    if mask_1.any():
        S_1 = torch.sqrt(1.0 + R[mask_1, 0, 0] - R[mask_1, 1, 1] - R[mask_1, 2, 2]) * 2
        q[mask_1, 0] = (R[mask_1, 2, 1] - R[mask_1, 1, 2]) / S_1
        q[mask_1, 1] = 0.25 * S_1
        q[mask_1, 2] = (R[mask_1, 0, 1] + R[mask_1, 1, 0]) / S_1
        q[mask_1, 3] = (R[mask_1, 0, 2] + R[mask_1, 2, 0]) / S_1
    
    # Case 3: R[1,1] > R[2,2]
    if mask_2.any():
        S_2 = torch.sqrt(1.0 + R[mask_2, 1, 1] - R[mask_2, 0, 0] - R[mask_2, 2, 2]) * 2
        q[mask_2, 0] = (R[mask_2, 0, 2] - R[mask_2, 2, 0]) / S_2
        q[mask_2, 1] = (R[mask_2, 0, 1] + R[mask_2, 1, 0]) / S_2
        q[mask_2, 2] = 0.25 * S_2
        q[mask_2, 3] = (R[mask_2, 1, 2] + R[mask_2, 2, 1]) / S_2
    
    # Case 4: R[2,2] is largest diagonal
    if mask_3.any():
        S_3 = torch.sqrt(1.0 + R[mask_3, 2, 2] - R[mask_3, 0, 0] - R[mask_3, 1, 1]) * 2
        q[mask_3, 0] = (R[mask_3, 1, 0] - R[mask_3, 0, 1]) / S_3
        q[mask_3, 1] = (R[mask_3, 0, 2] + R[mask_3, 2, 0]) / S_3
        q[mask_3, 2] = (R[mask_3, 1, 2] + R[mask_3, 2, 1]) / S_3
        q[mask_3, 3] = 0.25 * S_3
    
    return q
# def quaternion_multiply(q1, q2):
#     """
#     Quaternion multiplication
#     Args:
#         q1: first quaternion [b,v,r,srf,spp, 4]
#         q2: second quaternion [b,v,1,srf,spp, 4]
#     Returns:
#         q: result quaternion [b,v,r,srf,spp, 4]
#     """

#     # out = torch.zeros((q1.shape[0],q1.shape[1],q1.shape[2],q1.shape[3],q1.shape[4], 4), device=q1.device)
#     # for i in range(q1.shape[0]):
#     #     for j in range(q1.shape[1]):
#     #         Q1 = q1[i,j,:]
#     #         Q2 = q2[i,j]

#     w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
#     w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
#     return torch.stack([w, x, y, z], dim=1)

def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication
    Args:
        q1: first quaternion [b,v,r,srf,spp, 4]
        q2: second quaternion [b,v,1,srf,spp, 4]
    Returns:
        q: result quaternion [b,v,r,srf,spp, 4]
    """
    # Adjust dimensions for broadcasting
    # Separate quaternion components from last dimension
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Perform quaternion multiplication.
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Stack results along last dimension
    return torch.stack([w, x, y, z], dim=-1)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )



def build_rotation(r):
    #print('rotation shape is ',r.shape)
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    #print('correct')
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    #print('s',s.shape)
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = 0
    #print('R',R.shape)
    #print('L',L.shape)
    L = R @ L
    return L
