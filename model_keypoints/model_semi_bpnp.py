import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from .image_backbone import FeaturePyramid, ImageBackbone
from .point_backbone import PointBackbone 

from .utils import get_2d3d_node_correspondences, patchify
from vision3d.ops import knn_interpolate_pack_mode
from .match_utils import pairwiseL2Dist, RegularisedTransport, ransac_p3p
from .nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP

from model_keypoints.model_kpts_learning import create_model

class SemiBlindPnPSolver(nn.Module):
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
        self.kpts_3d_proj = nn.Linear(128, 128)
        self.kpts_2d_proj = nn.Linear(128, 128)

        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)
        self.ransac_p3p = ransac_p3p
        self.wbpnp = NonlinearWeightedBlindPnP()

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()

        # 1. using 3d keypoints learning model 
        #    which is pre-trained from train_anyscene_kpts.py
        output_dict = self.model_kpts(data_dict)

        # 2. unpack important learning results
        '''
            note-0108:
            due to the gpu memory limitation, we select top-k 3d keypoints,
                instead of only using a threshold
        '''
        intr_mat = data_dict["intrinsics"].detach()  # [3,3]
        transform = data_dict["transform"].detach()  # [4,4]
        kpts_3d_scores = output_dict["coarse_kpts"]
        kpts_3d_mask = (kpts_3d_scores[:,0] >= 0.40) # 0.35
        
        if torch.sum(kpts_3d_mask) >= 6666:
            _, indices = torch.sort(kpts_3d_scores[:,0], descending=True)
            kpts_3d_mask = indices[:6666]

        kpts_3d_pts = output_dict["kpts_3d_pts"][kpts_3d_mask,:] # [n,3]
        kpts_3d_fea = output_dict["kpts_3d_fea"][kpts_3d_mask,:] # [n,c]
        kpts_3d_pix = output_dict["kpts_3d_pix"][kpts_3d_mask,:] # [n,2]
        kpts_2d_pix = output_dict["kpts_2d_pix"] # [m,2]
        kpts_2d_fea = output_dict["kpts_2d_fea"] # [m,c]
        output_dict["kpts_3d_pix_selected"] = kpts_3d_pix
        
        # computing bearing vector of 2d pixels
        m = kpts_2d_fea.size(0)
        tmp_vector = torch.ones((m,3)).cuda()
        tmp_vector[:,:2] = kpts_2d_pix[:,:2]     # [m,3]
        inv_intr_mat = torch.inverse(intr_mat)   # [3,3]
        kpts_2d_bearing = torch.matmul(tmp_vector, inv_intr_mat.t()) # [m,3]

        # 2b. add random rotation and translation noise
        '''
            why add rigid-transformation noise?

            because we extract image and point cloud from a rgb-d image,
            in this case, the original transformation is an 4x4 identity matrix,
            which is easier to learn.

            so, for the fair comparision, we add transformation noise.
        '''
        pass

        # 3. feature extraction for semi-blind-pnp
        alpha = 0.05
        kpts_3d_fea = kpts_3d_fea + alpha*self.kpts_3d_proj(kpts_3d_fea) # [n,c]
        kpts_2d_fea = kpts_2d_fea + alpha*self.kpts_2d_proj(kpts_2d_fea) # [m,c]

        # 4. get correspondences from sinkhorn layer
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

        # 4b. add diffusion model for correspondence learning
        pass

        # 5. compute pose loss
        '''
            note-0108: we cannot use wbpnp loss, 
                beacuase of the memory burden is too large :(
        '''
        is_able = True
        if is_able:
            theta, theta0 = None, None
            p3d = kpts_3d_pts.unsqueeze(0)     # [1,n,3]
            p2d = kpts_2d_bearing[:,:2].unsqueeze(0) # [1,m,2]
            # tmp = p2d[0,:,0]
            # p2d[0,:,0] = p2d[0,:,1]
            # p2d[0,:,1] = tmp

            # using RANSAC (does not know it has problem?)
            theta0 = self.ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d)
            print("theta0: ", theta0)
            print("transform")
            print(transform)
            assert 1==-1
            
            # Run Weighted BPnP Optimization:
            p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
            p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
            theta = self.wbpnp(P, p2d_bearings, p3d, theta0)
            output_dict["theta0"] = theta0
            output_dict["theta"] = theta

        torch.cuda.synchronize()
        duration = time.time() - start_time
        print("cost time: ", duration)
        output_dict["duration"] = duration
        return output_dict

def create_solver_model(cfg):
    model_solver = SemiBlindPnPSolver(cfg)
    return model_solver