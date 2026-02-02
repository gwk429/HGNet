import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.SE3 import transform
from utils.pointcloud import make_point_cloud, estimate_normal
import open3d as o3d


def icp_my(src_keypts, tgt_keypts, pred_trans):
    """
    ICP algorithm to refine the initial transformation
    Input:
        - src_keypts [1, num_corr, 3] FloatTensor
        - tgt_keypts [1, num_corr, 3] FloatTensor
        - pred_trans [1, 4, 4] FloatTensor, initial transformation
    """
    trans = torch.empty(0, 4, 4).cuda()
    for i in range(src_keypts.size(0)):
        src_pcd = make_point_cloud(src_keypts.detach().cpu().numpy()[i])
        tgt_pcd = make_point_cloud(tgt_keypts.detach().cpu().numpy()[i])
        initial_trans = pred_trans[i].detach().cpu().numpy()
        # change the convension of transforamtion because open3d use left multi.
        refined_T = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, 0.10, initial_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
        refined_T = torch.from_numpy(refined_T[None, :, :]).to(pred_trans.device).float()
        trans = torch.cat((trans, refined_T), dim=0)
    warp_src_keypts = transform(src_keypts, trans)

    return warp_src_keypts


def space_normal(points, fixed_index=0):
    B, N, _ = points.shape  # 获取批次大小 B 和点的数量 N
    # 获取固定点的位置
    fixed_point = points[:, fixed_index, :].unsqueeze(1)  # [B, 1, 3]
    # 计算每两个点与固定点的向量 v1 和 v2
    v1 = points - fixed_point  # [B, N, 3] - [B, 1, 3] => [B, N, 3]
    # 计算叉积 (v1 与 v2)
    v1_expanded = v1.unsqueeze(2)  # [B, N, 3, 1]，用于广播
    v2_expanded = v1.unsqueeze(1)  # [B, 1, N, 3]，用于广播
    normal = torch.cross(v1_expanded, v2_expanded, dim=3)  # [B, N, N, 3]，法向量
    # # 计算法向量的模长 [B, N, N]
    # normal_norm = torch.norm(normal, dim=3)  # [B, N, N]
    # # 避免除以零
    # norm_mask = normal_norm > 1e-8  # 确保法向量的模长不为零
    # normal_norm = torch.where(norm_mask, normal_norm, torch.tensor(1e-8).cuda())
    # # 计算法向量的夹角余弦值 (dot(normal, normal) / (norm * norm))
    # dot_product = torch.sum(normal * normal, dim=3)  # [B, N, N]，法向量点积
    # cos_theta = dot_product / (normal_norm ** 2)  # 余弦值
    # # 防止cos_theta的数值溢出到 [-1, 1] 范围之外
    # cos_theta = torch.clamp(cos_theta, -1 + 1e-8, 1 - 1e-8)
    # # 计算夹角（弧度）
    # angle = torch.acos(cos_theta)
    # angle = torch.abs(angle * (180.0 / np.pi))
    # # 存储夹角
    # angles[:, :, :] = angle

    return normal


def compute_angle(normals1, normals2):
    """
    Compute the angles between two sets of normal vectors.
    Input:
        - normals1: [B, N, N, 3] Tensor, first set of normal vectors
        - normals2: [B, N, N, 3] Tensor, second set of normal vectors
    Output:
        - angles: [B, N, N] Tensor, angles between corresponding normal vectors
    """
    # 计算点积 (dot product)
    dot_product = torch.sum(normals1 * normals2, dim=-1)  # [B, N, N]

    # 计算每个法向量的模 (norm)
    norm1 = torch.norm(normals1, dim=-1)  # [B, N, N]
    norm2 = torch.norm(normals2, dim=-1)  # [B, N, N]

    # 计算余弦值
    cos_theta = dot_product / (norm1 * norm2 + 1e-8)  # 防止除以零，加入小的 epsilon

    # 限制余弦值范围在 [-1, 1] 之间，以防数值误差
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # 计算夹角 (弧度)
    angles = torch.acos(cos_theta)  # [B, N, N]

    angles = angles * 180.0 / torch.pi  # 转换为角度

    # 将角度限制在 0 到 90 度之间
    angles = torch.minimum(angles, 180.0 - angles)

    return angles


#
#
# x = torch.randn(16, 1000, 3)
# y = space_normal(x)

def darboux(normals):
    # normals = torch.nn.functional.normalize(normals, dim=-1, p=2)
    normals = normals.permute(0, 2, 1)  #[8,3,1000]
    l1 = torch.norm(normals[:, :, None, :], p=2, dim=1).permute(0, 2, 1)  # [B, N, 1]
    # l2 = torch.norm(normals_knn, p=2, dim=1)  # [B, N, K]
    a3 = torch.sum(normals[:, :, :, None] * normals[:, :, None, :], dim=1) / (l1 + 1e-10)  # [B, N, K]
    epsilon = 1e-8
    angle_radians = torch.acos(torch.clamp(a3, -1 + epsilon, 1 - epsilon))
    a3 = torch.abs(angle_radians * (180.0 / np.pi))
    return a3


def Space_normals(points, top2):
    B, N, _ = points.shape  # 获取批次大小 B 和点的数量 N

    # 使用 top2_seeds 中的索引从 src_keypts 中提取对应的坐标
    top2_keypts = points[torch.arange(B).unsqueeze(1), top2]  # [B, 2, 3]

    # 将提取的两个坐标分别存放到两个变量中
    fixed_p1 = top2_keypts[:, 0, :].unsqueeze(1)  # 每个批次第一个点的坐标，形状 [B, 1, 3]
    fixed_p2 = top2_keypts[:, 1, :].unsqueeze(1)  # 每个批次第二个点的坐标，形状 [B, 1, 3]

    # 计算每两个点与固定点之间的向量 v1 和 v2
    v1 = points - fixed_p1  # [B, N, 3] - [B, 1, 3] => [B, N, 3]
    v2 = points - fixed_p2  # [B, N, 3] - [B, 1, 3] => [B, N, 3]

    # 计算叉积 (v1 与 v2)，得到法向量
    normals = torch.cross(v1, v2, dim=2)  # [B, N, 3]，法向量

    normals_unit = normals / (normals.norm(p=2, dim=-1, keepdim=True) + 1e-10)  # [B, N, 3]
    a3 = torch.matmul(normals_unit, normals_unit.transpose(1, 2))

    epsilon = 1e-8
    angle_radians = torch.acos(torch.clamp(a3, -1 + epsilon, 1 - epsilon))
    a3 = torch.abs(angle_radians * (180.0 / np.pi))
    a3 = torch.where(a3 > 90, 180 - a3, a3)
    return a3


def draw_registration_result(source, target, transformation, i):
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud(target)

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

    # 可视化并保存为 PNG
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    # 进入交互式模式
    vis.run()  # 运行交互式可视化，手动调整视角
    view_ctl = vis.get_view_control()
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    # 保存 PNG 文件
    path = f"F:/gwk/hyperreg/{i}-.png"
    plt.imsave(path, np.asarray(image))
