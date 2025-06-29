import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import cv2 as cv

from torch.nn.parallel import DistributedDataParallel
from vision3d.utils.logger import get_logger
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render)

from vision3d.models.point_transformer import PointTransformerBlock
from vision3d.layers.pointnet import PNConv
from vision3d.ops.so3 import axis_angle_to_rotation_matrix
from .match_utils import pairwiseL2Dist, RegularisedTransport
from model_keypoints.model_kpts_learning import create_model
from loss.loss_exp import show_pcd

class BlindPnPNeuralSolver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d

        '''
            load pre-train model and set it as evaluation mode
        '''
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos_kpts/"
        # ckpt_path = base_path + "epoch-35.pth"
        # ckpt_path = base_path + "epoch-25.pth"
        ckpt_path = base_path + "epoch-25-kitti.pth"

        self.model_kpts = create_model(cfg)
        self.model_kpts = self.model_kpts#.cuda().eval()
        
        logger = get_logger()
        logger.info("Loading checkpoint from '{}'.".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

        if isinstance(self.model_kpts, DistributedDataParallel):
            self.model_kpts = self.model_kpts.module
        missing_keys, unexpected_keys = self.model_kpts.load_state_dict(state_dict['model'], strict=False)
        
        if len(missing_keys) > 0:
            logger.warn(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warn(f"Unexpected keys: {unexpected_keys}")

        '''
            a tiny neural network in semi-blind-pnp solver
        '''
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)

        self.kpts_2d_pre = nn.Conv1d(3, 512, kernel_size=1)
        self.kpts_3d_pre = nn.Conv1d(3, 512, kernel_size=1)
        self.kpts_2d_net_1 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.kpts_3d_net_1 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.kpts_2d_net_2 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.kpts_3d_net_2 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.kpts_2d_net_3 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.kpts_3d_net_3 = PointTransformerBlock(512,512,512,num_neighbors=16)
        self.so_pred_net = nn.Linear(512*2, 6)

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()

        # 1. using 3d keypoints learning model 
        #    which is pre-trained from train_anyscene_kpts.py
        with torch.no_grad():
            output_dict = self.model_kpts(data_dict)

        # 2. unpack important learning results
        intr_mat = data_dict["intrinsics"].detach()  # [3,3]
        transform = data_dict["transform"].detach()  # [4,4]
        kpts_3d_scores = output_dict["coarse_kpts"]
        kpts_3d_mask = (kpts_3d_scores[:,0] >= 0.40) # 0.35

        kpts_3d_pts = output_dict["kpts_3d_pts"][kpts_3d_mask,:] # [n,3]
        kpts_3d_pix = output_dict["kpts_3d_pix"][kpts_3d_mask,:] # [n,2]
        kpts_2d_pts = output_dict["kpts_2d_pts"] # [m,3]
        kpts_2d_pix = output_dict["kpts_2d_pix"] # [m,2]
        output_dict["kpts_3d_pts"] = kpts_3d_pts
        output_dict["kpts_3d_pix"] = kpts_3d_pix

        # print("==> matching status: ", 
        #     " pts 3d: ", kpts_3d_pts.size(0), " pts 2d: ", kpts_2d_pts.size(0))
        n = kpts_3d_pts.size(0)
        if n >= 500:
            kpts_3d_pts = kpts_3d_pts[::4,:]
            kpts_3d_pix = kpts_3d_pix[::4,:]
            output_dict["kpts_3d_pts"] = kpts_3d_pts
            output_dict["kpts_3d_pix"] = kpts_3d_pix
        m = kpts_2d_pts.size(0)
        if m >= 500:
            kpts_2d_pts = kpts_2d_pts[::2,:]
            kpts_2d_pix = kpts_2d_pix[::2,:]
            output_dict["kpts_2d_pts"] = kpts_2d_pts
            output_dict["kpts_2d_pix"] = kpts_2d_pix
        # report
        # print("==> now matching status: ", 
        #     " pts 3d: ", kpts_3d_pts.size(0), " pts 2d: ", kpts_2d_pts.size(0))

        # 3. optimization layer with blind pnp layer
        '''
            As 2d and 3d keypoints have many inliers, we approximate 
                2d-3d blind pnp problem as a 
                2d-2d point set registration problem on an image plane
            
            it is based on one assumptiont that the transformation matrix is 
                closed to an identity matrix (statified in most of cases)
        '''
        ## 3.1 project 2d and 3d kpts on an image plane
        #       compute bearing vectors from 2d pixels and 3d points
        m = kpts_2d_pix.size(0)
        tmp_vector = torch.ones((m,3)).cuda()
        tmp_vector[:,0] = kpts_2d_pix[:,1]
        tmp_vector[:,1] = kpts_2d_pix[:,0]     # [m,3]
        inv_intr_mat = torch.inverse(intr_mat)   # [3,3]
        kpts_2d_bea = torch.matmul(inv_intr_mat, tmp_vector.t()).t() # [m,3]

        kpts_2d_nbv = kpts_2d_bea
        kpts_3d_nbv = kpts_3d_pts/kpts_3d_pts[:,2:3]
        kpts_2d_nbv = torch.nn.functional.normalize(kpts_2d_nbv, p=2, dim=-1)
        kpts_3d_nbv = torch.nn.functional.normalize(kpts_3d_nbv, p=2, dim=-1)

        '''
            a check for 3d points and its pixels
        '''
        # kpts_3d_pix = render(kpts_3d_pts, intr_mat)
        # print("kpts_3d_pix === after")
        # assert 1==-1

        ## 3.2 point set feature learning with point transformer
        #       approximate the transformation with 
        #           a series of se(3) matrix + B_3 projection
        kpts_2d_xyz = kpts_2d_nbv.t().unsqueeze(0)  # [m,3] => [1,3,m]
        kpts_3d_xyz = kpts_3d_nbv.t().unsqueeze(0)  # [n,3] => [1,3,n]
        kpts_2d_fea = self.kpts_2d_pre(kpts_2d_xyz)
        kpts_3d_fea = self.kpts_3d_pre(kpts_3d_xyz)

        kpts_2d_fea, _ = self.kpts_2d_net_1(kpts_2d_fea, kpts_2d_xyz) # [1,c,m]
        kpts_3d_fea, _ = self.kpts_2d_net_1(kpts_3d_fea, kpts_3d_xyz) # [1,c,n]
        kpts_2d_fea, _ = self.kpts_2d_net_2(kpts_2d_fea, kpts_2d_xyz) # [1,c,m]
        kpts_3d_fea, _ = self.kpts_2d_net_2(kpts_3d_fea, kpts_3d_xyz) # [1,c,n]

        mean_kpts_2d_fea = torch.mean(kpts_2d_fea, dim=2) # [1,c]
        mean_kpts_3d_fea = torch.mean(kpts_3d_fea, dim=2) # [1,c]
        pose_fea = torch.concatenate((mean_kpts_2d_fea, mean_kpts_3d_fea), dim=1) # [1,2c]
        pose = self.so_pred_net(pose_fea[0]) # [6]
        R_pd = axis_angle_to_rotation_matrix(pose[0:3]) # [3,3]
        T_pd = pose[3:6].unsqueeze(0)                   # [1,3]
        tmp = torch.matmul(kpts_2d_nbv, R_pd) + T_pd    # [m,3]
        tmp = torch.nn.functional.normalize(tmp, p=2, dim=-1)
        kpts_2d_xyz = tmp.t().unsqueeze(0)              # [1,3,m]

        # kpts_2d_fea, _ = self.kpts_2d_net_2(kpts_2d_fea, kpts_2d_xyz) # [1,c,m]
        # kpts_3d_fea, _ = self.kpts_2d_net_2(kpts_3d_fea, kpts_3d_xyz) # [1,c,n]

        # mean_kpts_2d_fea = torch.mean(kpts_2d_fea, dim=2) # [1,c]
        # mean_kpts_3d_fea = torch.mean(kpts_3d_fea, dim=2) # [1,c]
        # pose_fea = torch.concatenate((mean_kpts_2d_fea, mean_kpts_3d_fea), dim=1) # [1,2c]
        # pose = self.so_pred_net(pose_fea[0]) # [6]
        # R_pd = axis_angle_to_rotation_matrix(pose[0:3]) # [3,3]
        # T_pd = pose[3:6].unsqueeze(0)                   # [1,3]
        # tmp = torch.matmul(kpts_2d_nbv, R_pd) + T_pd    # [m,3]
        # tmp = torch.nn.functional.normalize(tmp, p=2, dim=-1)
        # kpts_2d_xyz = tmp.t().unsqueeze(0)              # [1,3,m]

        # kpts_2d_fea, _ = self.kpts_2d_net_3(kpts_2d_fea, kpts_2d_xyz) # [1,c,m]
        # kpts_3d_fea, _ = self.kpts_2d_net_3(kpts_3d_fea, kpts_3d_xyz) # [1,c,n]

        # mean_kpts_2d_fea = torch.mean(kpts_2d_fea, dim=2) # [1,c]
        # mean_kpts_3d_fea = torch.mean(kpts_3d_fea, dim=2) # [1,c]
        # pose_fea = torch.concatenate((mean_kpts_2d_fea, mean_kpts_3d_fea), dim=1) # [1,2c]
        # pose = self.so_pred_net(pose_fea[0]) # [6]
        # R_pd = axis_angle_to_rotation_matrix(pose[0:3]) # [3,3]
        # T_pd = pose[3:6].unsqueeze(0)                   # [1,3]
        # tmp = torch.matmul(kpts_2d_nbv, R_pd) + T_pd    # [m,3]
        # tmp = torch.nn.functional.normalize(tmp, p=2, dim=-1)
        # kpts_2d_xyz = tmp.t().unsqueeze(0)              # [1,3,m]

        output_dict["kpts_3d_xyz"] = kpts_3d_xyz        # [1,3,n]
        output_dict["kpts_2d_xyz"] = kpts_2d_xyz        # [1,3,m]

        # 4. 2d and 3d keypoints visulization analysis
        '''
            visulization analysis --- ok 
                kpts_3d_pix and kpts_2d_pix computation --- ok
                kpts_3d_pts and kpts_2d_pts computation --- ok
        '''
        is_need_vis_2d = False
        if is_need_vis_2d:
            rgb_image = data_dict["image"].detach().cpu().numpy()[0]
            kpts_3d_pix_vis = kpts_3d_pix.detach().cpu().numpy()
            for i in range(kpts_3d_pix_vis.shape[0]):
                x = kpts_3d_pix_vis[i,1]
                y = kpts_3d_pix_vis[i,0]
                cv.circle(rgb_image, (int(x), int(y)), radius=3, color=(255,0,0), thickness=2)
            kpts_2d_pix_vis = kpts_2d_pix.detach().cpu().numpy()
            for i in range(kpts_2d_pix_vis.shape[0]):
                x = kpts_2d_pix_vis[i,1]
                y = kpts_2d_pix_vis[i,0]
                cv.circle(rgb_image, (int(x), int(y)), radius=2, color=(0,255,0), thickness=2)

            cv.imshow("rgb_image", rgb_image)
            cv.waitKey(0)
            print("rgb_image: ", rgb_image.shape)
            # assert 1==-1

        is_need_vis_3d = False
        if is_need_vis_3d:
            # pts_2d = kpts_2d_pts.detach().cpu().numpy()
            # pts_3d = kpts_3d_pts.detach().cpu().numpy()
            # pts_2d_bn = kpts_2d_nbv.detach().cpu().numpy()
            pts_2d_bn = kpts_2d_xyz[0].t().detach().cpu().numpy()
            pts_3d_bn = kpts_3d_nbv.detach().cpu().numpy()
            pts_2d = pts_2d_bn
            pts_3d = pts_3d_bn

            pcd_2d = o3d.geometry.PointCloud()
            pcd_2d.points = o3d.utility.Vector3dVector(pts_2d[:,:3])
            colors = np.zeros_like(pts_2d[:,:3])
            colors[:,1] = 1.0 # green
            pcd_2d.colors = o3d.utility.Vector3dVector(colors[:,:3])
            # pcd_2d.normals = o3d.utility.Vector3dVector(pts_2d_sn[:,:3])

            pcd_3d = o3d.geometry.PointCloud()
            pcd_3d.points = o3d.utility.Vector3dVector(pts_3d[:,:3])
            colors = np.zeros_like(pts_3d[:,:3])
            colors[:,2] = 1.0 # blue
            pcd_3d.colors = o3d.utility.Vector3dVector(colors[:,:3])
            # pcd_3d.normals = o3d.utility.Vector3dVector(pts_3d_sn[:,:3])
            # show_pcd(pcd_2d)
            # show_pcd(pcd_3d)
            show_pcd(pcd_3d+pcd_2d)
            # o3d.visualization.draw_geometries([pcd_3d+pcd_2d],
            #     point_show_normal=True)
            # assert 1==-1

        torch.cuda.synchronize()
        duration = time.time() - start_time
        # print("cost time (all bpnp): ", duration)
        output_dict["duration"] = duration
        return output_dict

def create_solver_model(cfg):
    model_solver = BlindPnPNeuralSolver(cfg)
    return model_solver