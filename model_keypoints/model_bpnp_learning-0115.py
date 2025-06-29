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

from .image_backbone import ImageBackbone
from .point_backbone import PointBackbone 

from .utils import get_2d3d_node_correspondences, patchify
from vision3d.ops import knn_interpolate_pack_mode
from .match_utils import pairwiseL2Dist, RegularisedTransport, ransac_p3p
from .nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP

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
        ckpt_path = base_path + "epoch-35.pth"
        # ckpt_path = base_path + "epoch-25.pth"

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
        self.wbpnp = NonlinearWeightedBlindPnP()
        self.ransac_p3p = ransac_p3p
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)

        self.mlp_image = nn.Sequential(
            nn.Linear(6  , 64), nn.Sigmoid(), nn.Linear(64 ,128), nn.Sigmoid(),
            nn.Linear(128,128), nn.Sigmoid())
        self.mlp_point = nn.Sequential(
            nn.Linear(6  , 64), nn.Sigmoid(), nn.Linear(64 ,128), nn.Sigmoid(),
            nn.Linear(128,128), nn.Sigmoid())

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
        output_dict["kpts_3d_pix_selected"] = kpts_3d_pix

        # 3. optimization layer with surface normal guided blind pnp layer
        '''
            is there a possibility to online minimize the distribution 
                between 2d and 3d keypoints

            as 2d and 3d keypoints features are not very reliable, 
                we notice that 2d and 3d inliers keypoint ratio are higher
                such as 2d inlier ratio 2931/3310 = 88.5% (contains one-to-many)
                        3d inlier ratio 2931/3413 = 85.8%
            
            so, maybe there is some potential matching mechnism...
        '''
        ## 3.1 using surface normal to remove unaligned 2d-3d keypoints
        images_mask = data_dict["images_mask"][:,:,0].detach()
        images_mask = images_mask.bool()
        kpts_2d_pts_normal = data_dict["surface_normal_2d"][images_mask,:]  # [m,3]
        kpts_3d_pts_normal = data_dict["surface_normal_3d"][kpts_3d_mask,:] # [n,3]
        kpts_2d_pts_normal = torch.nn.functional.normalize(kpts_2d_pts_normal, p=2, dim=-1)
        kpts_3d_pts_normal = torch.nn.functional.normalize(kpts_3d_pts_normal, p=2, dim=-1)
        
        mat_primitive = pairwiseL2Dist(kpts_2d_pts_normal.unsqueeze(0), kpts_3d_pts_normal.unsqueeze(0)) #[m,n] 
        mat_primitive = mat_primitive[0]
        kpts_3d_sc, _ = torch.min(mat_primitive, dim=0)
        kpts_2d_sc, _ = torch.min(mat_primitive, dim=1)
        mask_3d = (kpts_3d_sc >= 0.01) # 0.0001
        mask_2d = (kpts_2d_sc >= 0.01)

        # kpts_2d_pix = kpts_2d_pix[~mask_2d]
        # kpts_3d_pix = kpts_3d_pix[~mask_3d]
        # kpts_2d_pts = kpts_2d_pts[~mask_2d]
        # kpts_3d_pts = kpts_3d_pts[~mask_3d]
        # kpts_2d_pts_normal = kpts_2d_pts_normal[~mask_2d]
        # kpts_3d_pts_normal = kpts_3d_pts_normal[~mask_3d]

        # maximum correspondences
        max_2d_pts = 500
        if kpts_2d_pix.size(0) > max_2d_pts:
            kpts_2d_pix = kpts_2d_pix[::2, :]
            kpts_2d_pts = kpts_2d_pts[::2, :]
            kpts_2d_pts_normal = kpts_2d_pts_normal[::2, :]
        if kpts_3d_pix.size(0) >= max_2d_pts:
            kpts_3d_pix = kpts_3d_pix[::2, :]
            kpts_3d_pts = kpts_3d_pts[::2, :]
            kpts_3d_pts_normal = kpts_3d_pts_normal[::2, :]

        output_dict["kpts_2d_pix"] = kpts_2d_pix
        output_dict["kpts_3d_pix"] = kpts_3d_pix

        # print("kpts_2d_pts_normal: ", kpts_2d_pts_normal.size())
        # print("kpts_3d_pts_normal: ", kpts_3d_pts_normal.size())
        # assert 1==-1

        ## 3.2 using blind pnp layers
        # sub-step one --- compute normalized bearing vector of 2d pixels and 3d points
        m = kpts_2d_pix.size(0)
        tmp_vector = torch.ones((m,3)).cuda()
        tmp_vector[:,:2] = kpts_2d_pix[:,:2]     # [m,3]
        inv_intr_mat = torch.inverse(intr_mat)   # [3,3]
        kpts_2d_bea = torch.matmul(tmp_vector, inv_intr_mat.t()) # [m,3]
        tmp = kpts_2d_bea[:,0].clone()
        kpts_2d_bea[:,0] = kpts_2d_bea[:,1]
        kpts_2d_bea[:,1] = tmp
        kpts_2d_nbv = torch.nn.functional.normalize(kpts_2d_bea, p=2, dim=-1)
        kpts_3d_nbv = torch.nn.functional.normalize(kpts_3d_pts, p=2, dim=-1)

        # sub-step two --- concate surface normal and normalized bearing vector features
        kpts_2d_fea = torch.concatenate((kpts_2d_pts_normal, kpts_2d_nbv), dim=1)
        kpts_3d_fea = torch.concatenate((kpts_3d_pts_normal, kpts_3d_nbv), dim=1)

        # sub-step three --- deep feature learning
        kpts_2d_fea = self.mlp_image(kpts_2d_fea)
        kpts_3d_fea = self.mlp_point(kpts_3d_fea)

        # sub-step four --- compute matching matrix
        kpts_3d_fea = kpts_3d_fea.unsqueeze(0) # [1,n,c]
        kpts_2d_fea = kpts_2d_fea.unsqueeze(0) # [1,m,c]
        f3d = torch.nn.functional.normalize(kpts_3d_fea, p=2, dim=-1)
        f2d = torch.nn.functional.normalize(kpts_2d_fea, p=2, dim=-1)
        M = pairwiseL2Dist(f2d, f3d) # [m,n]
        '''
            P is a probability matrix that torch.sum(P)=1.0
        '''
        b, m, n = M.size()
        num_points_3d = n
        num_points_2d = m
        r = M.new_zeros((b, m)) # bxm
        c = M.new_zeros((b, n)) # bxn
        for i in range(b):
            r[i, :m] = 1.0 / m
            c[i, :n] = 1.0 / n
        P = self.sinkhorn(M, r, c)
        output_dict["P"] = P

        # sub-step five --- compute pose and correspondence loss
        is_able = False
        if is_able:
            theta, theta0 = None, None
            p3d = kpts_3d_pts.unsqueeze(0)     # [1,n,3]
            p2d = kpts_2d_bea[:,:2].unsqueeze(0) # [1,m,2]
            # tmp = p2d[0,:,0]
            # p2d[0,:,0] = p2d[0,:,1]
            # p2d[0,:,1] = tmp

            # using RANSAC (does not know it has problem?)
            theta0 = self.ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d)
            # print("theta0: ", theta0)
            # print("transform")
            # print(transform)
            # assert 1==-1
            
            # Run Weighted BPnP Optimization:
            p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
            p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
            theta = self.wbpnp(P, p2d_bearings, p3d, theta0)
            output_dict["theta0"] = theta0
            output_dict["theta"] = theta

        # 4. 2d and 3d keypoints visulization analysis
        '''
            visulization analysis --- ok 
                kpts_3d_pix and kpts_2d_pix computation --- ok
                kpts_3d_pts and kpts_2d_pts computation --- ok
        '''
        is_need_vis_2d = True
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

        is_need_vis_3d = True
        if is_need_vis_3d:
            pts_2d = kpts_2d_pts.detach().cpu().numpy()
            pts_3d = kpts_3d_pts.detach().cpu().numpy()
            pts_2d_sn = kpts_2d_pts_normal.detach().cpu().numpy()
            pts_3d_sn = kpts_3d_pts_normal.detach().cpu().numpy()
            pts_2d_bn = kpts_2d_nbv.detach().cpu().numpy()
            pts_3d_bn = kpts_3d_nbv.detach().cpu().numpy()
            pts_2d = pts_2d_bn
            pts_3d = pts_3d_bn

            pcd_2d = o3d.geometry.PointCloud()
            pcd_2d.points = o3d.utility.Vector3dVector(pts_2d[:,:3])
            colors = np.zeros_like(pts_2d[:,:3])
            colors[:,2] = 1.0
            pcd_2d.colors = o3d.utility.Vector3dVector(colors[:,:3])
            # pcd_2d.normals = o3d.utility.Vector3dVector(pts_2d_sn[:,:3])

            pcd_3d = o3d.geometry.PointCloud()
            pcd_3d.points = o3d.utility.Vector3dVector(pts_3d[:,:3])
            colors = np.zeros_like(pts_3d[:,:3])
            colors[:,1] = 1.0
            pcd_3d.colors = o3d.utility.Vector3dVector(colors[:,:3])
            # pcd_3d.normals = o3d.utility.Vector3dVector(pts_3d_sn[:,:3])
            # show_pcd(pcd_2d)
            # show_pcd(pcd_3d)
            show_pcd(pcd_3d+pcd_2d)
            # o3d.visualization.draw_geometries([pcd_3d+pcd_2d],
            #     point_show_normal=True)
            assert 1==-1

        torch.cuda.synchronize()
        duration = time.time() - start_time
        print("cost time (all bpnp): ", duration)
        output_dict["duration"] = duration
        return output_dict

def create_solver_model(cfg):
    model_solver = BlindPnPNeuralSolver(cfg)
    return model_solver