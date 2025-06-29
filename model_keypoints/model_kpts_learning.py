import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
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
from .match_utils import pairwiseL2Dist, RegularisedTransport

class baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim,
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation)

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim,
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma)

        self.coarse_matching = SuperPointMatchingMutualTopk(
            cfg.model.coarse_matching.num_correspondences,
            k=cfg.model.coarse_matching.topk,
            threshold=cfg.model.coarse_matching.similarity_threshold)
        
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)

    def unpack_2d_3d_data(self, data_dict, output_dict):
        '''
            a little change if the input is normal
            [B,480,640,3] => [B,1,480,640,3] => [B,3,480,640]
        '''
        # 2d image branch
        image = data_dict["image"].unsqueeze(1).detach() 
        image = image.transpose(1, -1)
        image = image.squeeze(-1)

        depth = data_dict["depth"].detach()  # (B, H, W)
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = data_dict["transform"].detach()

        img_h = image.shape[2]
        img_w = image.shape[3]
        img_h_f = img_h
        img_w_f = img_w
        output_dict["transform"] = transform
        output_dict["img_h_f"] = img_h_f
        output_dict["img_w_f"] = img_w_f

        img_points, img_masks = back_project(depth, intrinsics, depth_limit=6.0, transposed=True, return_mask=True)
        img_points = img_points.squeeze(0)  # (B, H, W, 3) -> (H, W, 3)
        img_masks = img_masks.squeeze(0)  # (B, H, W) -> (H, W)
        img_pixels = create_meshgrid(img_h, img_w).float()  # (H, W, 2)
        
        '''
            note-1227: img_points_fa is only used in loss function
        '''
        H, W = img_points.size(0), img_points.size(1)
        tmp_img_points = img_points.view(-1, 3) # (HxW, 3)
        tf = torch.inverse(transform)
        rr = tf[:3,0:3]
        tt = tf[:3,3:4]
        tmp = torch.matmul(rr, tmp_img_points.t())
        tmp = tmp + tt
        tmp_img_points_tune = tmp.t().view(H, W, 3) # (H, W, 3)
        output_dict["img_points_fa"] = tmp_img_points_tune

        img_points_f = img_points  # (H, H, 3)
        img_masks_f = img_masks  # (H, H)
        img_pixels_f = img_pixels  # (H, W, 2)
        output_dict["img_pixels_fa"] = img_pixels_f

        img_points = img_points.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels = img_pixels.view(-1, 2)  # (H, W, 2) -> (HxW, 2)
        img_masks  = img_masks.view(-1)  # (H, W) -> (HxW)
        img_points_f = img_points_f.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels_f = img_pixels_f.view(-1, 2)  # (H/2xW/2, 2)
        img_masks_f  = img_masks_f.view(-1)  # (H, W) -> (HxW)

        output_dict["img_points"] = img_points
        output_dict["img_pixels"] = img_pixels
        output_dict["img_masks"] = img_masks
        output_dict["img_points_f"] = img_points_f
        output_dict["img_pixels_f"] = img_pixels_f
        output_dict["img_masks_f"] = img_masks_f

        # 3d point cloud branch
        pcd_points = data_dict["points"][0].detach()
        pcd_points_f = data_dict["points"][0].detach()
        pcd_pixels_f = render(pcd_points_f, intrinsics, extrinsics=transform, rounding=False)

        output_dict["pcd_points"] = pcd_points
        output_dict["pcd_points_f"] = pcd_points_f
        output_dict["pcd_pixels_f"] = pcd_pixels_f
        return image, output_dict

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}
        
        # 1. Unpack data from data dict
        image, output_dict = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = data_dict["points_rgb"].detach()

        # 2. Backbone
        #    experiment report:
        #       load pre-trained backbone networks is very good :)
        img_feats_list = self.img_backbone(image)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)

        # 3. coarse 3d keypoints extraction
        #    experiment report:
        #      feature interaction between 2d and 3d kpts not good :(
        # TODO: add a circle loss
        images_mask = data_dict["images_mask"][:,:,0].detach()
        images_mask = images_mask.bool()
        kpts_2d_fea = img_feats_f[0][:,images_mask].transpose(1,0) # [n,c]
        kpts_3d_fea = pcd_feats_f                                  # [m,c]                         
        kpts_2d_pts = output_dict["img_points_fa"][images_mask,:]
        kpts_2d_pix = output_dict["img_pixels_fa"][images_mask,:]
        output_dict["kpts_2d_pix"] = kpts_2d_pix                   # [n,2]
        output_dict["kpts_2d_pts"] = kpts_2d_pts                   # [n,3]
        output_dict["kpts_3d_pts"] = output_dict["pcd_points_f"]   # [m,3]
        output_dict["kpts_3d_pix"] = output_dict["pcd_pixels_f"]   # [n,2]
        # assert 1==-1

        alpha = 1 # optimal parameter, aligned with loss_kpts.py
        kpts_2d_fea_n = kpts_2d_fea.unsqueeze(0)
        kpts_3d_fea_n = kpts_3d_fea.unsqueeze(0)
        kpts_2d_fea_n = torch.nn.functional.normalize(kpts_2d_fea_n, p=2, dim=-1)
        kpts_3d_fea_n = torch.nn.functional.normalize(kpts_3d_fea_n, p=2, dim=-1)
        mat_primitive = pairwiseL2Dist(kpts_2d_fea_n, kpts_3d_fea_n)  
        mat_primitive = torch.exp(-alpha*mat_primitive[0])  # [n,m]
        kpts_3d_scores, kpts_3d_indices = torch.max(mat_primitive, dim=0)
        kpts_3d_scores = kpts_3d_scores.unsqueeze(1)
        output_dict["coarse_kpts"] = kpts_3d_scores
        output_dict["kpts_2d_fea"] = kpts_2d_fea_n[0]
        output_dict["kpts_3d_fea"] = kpts_3d_fea_n[0]

        # 4. matching quality analysis with mat_primitive
        #    conclusions:
        #      coarse 3d kpts learning does not guarantee correspondence accuracy
        if False:
            mask_3x_gt = (data_dict["points_mask"][:,1] == 1)
            kpts_3d_mask = (kpts_3d_scores[:,0] >= 0.35) & mask_3x_gt
            kpts_2d_pixels = data_dict["kpts_2d_pixels"]
            kpts_pixels_3d = output_dict["pcd_pixels_f"][kpts_3d_mask,:]
            kpts_pixels_2d = kpts_2d_pixels[kpts_3d_indices[kpts_3d_mask],:]
            mean_pixel_err = torch.mean(torch.norm(kpts_pixels_3d - kpts_pixels_2d, dim=0))
            print("mean_pixel_err: ", mean_pixel_err)
        
        # 5. add blind pnp layers for correspondences learning
        #      a way to reference go-match (eccv-22)
        '''
            note-0107 this part is removed to model_semi_bpnp.py
        '''
        is_train_bpnp = True
        output_dict["is_train_bpnp"] = False
        if  is_train_bpnp:
            output_dict["is_train_bpnp"] = True
        
        torch.cuda.synchronize()
        duration = time.time() - start_time
        # print("cost time: ", duration)
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline(cfg)
    return model

