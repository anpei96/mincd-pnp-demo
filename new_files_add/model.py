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
    render,
)

# isort: split
from .fusion_module  import CrossModalFusionModule
from .image_backbone import FeaturePyramid, ImageBackbone
from .point_backbone import PointBackbone
from .base_model     import baseI2P
from .match_utils import pairwiseL2Dist, RegularisedTransport

from vision3d.models.point_transformer import PointTransformerBlock
from vision3d.layers.pointnet import PNConv
from vision3d.ops.so3 import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

class baseline(baseI2P):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        self.img_h_c = 24
        self.img_w_c = 32
        self.img_num_levels_c = 3
        self.overlap_threshold = 0.3
        self.pcd_min_node_size = 5

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

        self.transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding)

        self.img_pyramid = FeaturePyramid(cfg.model.transformer.output_dim)

        '''
            newly add for the auxiliary learning task
        '''
        self.kpts_2d_pre = nn.Conv1d(131, 128, kernel_size=1)
        self.kpts_3d_pre = nn.Conv1d(131, 128, kernel_size=1)
        self.kpts_net_2d = PointTransformerBlock(128,128,128,num_neighbors=16)
        self.kpts_net_3d = PointTransformerBlock(128,128,128,num_neighbors=16)
        self.so_pred_net = nn.Linear(128*2, 6)

        # flops
        # from thop import profile
        # input = torch.rand((6,512,512)).cuda()
        # flops, params = profile(self.img_backbone, inputs=(input,))
        # # stat(self.img_backbone, (6,512,512))
        # assert 1==-1

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}
        
        # 1. Unpack data from data dict
        image, output_dict = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = data_dict["points_rgb"].detach()
        pcd_feats = torch.concatenate((pcd_feats,
            data_dict["surface_normal_3d"].detach()), dim=1)
        normal_img = data_dict["surface_normal_2d"].detach() 
        normal_img = normal_img.transpose(2,0).transpose(2,1).unsqueeze(0)
        image = torch.concatenate((image,normal_img), dim=1)

        # 2. Backbone
        img_feats_list = self.img_backbone(image)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)

        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")

        # 3. Transformer
        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)

        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0),
        )

        # 3.3 Post-transformer image feature pyramid
        img_feats_c = img_feats_c.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        all_img_feats_c = self.img_pyramid(img_feats_c)
        all_img_feats_c = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c]
        img_feats_c = torch.cat(all_img_feats_c, dim=0)

        # 4. Coarse-level matching
        pcd_feats_c = pcd_feats_c.squeeze(0)
        img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        output_dict["img_feats_c"] = img_feats_c
        output_dict["pcd_feats_c"] = pcd_feats_c
        output_dict = self.genenrate_label(output_dict)
       
        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        '''
            note-anpei-0126

            learning correspondences from blind pnp perspective using 
                triplet optimization approximation
            in this scheme, the proposed scehme serves as a multi-task
                learning mechnisim
        '''
        if self.training:
            '''
                extra step 1. 3d keypoints learning
            '''
            images_mask = data_dict["images_mask"][:,:,0].detach()
            images_mask = images_mask.bool()
            kpts_2d_fea = img_feats_list[0][0][:,images_mask].transpose(1,0) # [n,c]
            kpts_3d_fea = pcd_feats_list[0]                                  # [m,c]                         
            kpts_2d_pts = output_dict["img_points_fa"][images_mask,:]
            kpts_2d_pix = output_dict["img_pixels_fa"][images_mask,:]
            output_dict["kpts_2d_pix"] = kpts_2d_pix                   # [n,2]
            output_dict["kpts_2d_pts"] = kpts_2d_pts                   # [n,3]
            output_dict["kpts_3d_pts"] = output_dict["pcd_points_f"]   # [m,3]
            output_dict["kpts_3d_pix"] = output_dict["pcd_pixels_f"]   # [n,2]

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

            '''
                extra step 2. pose estimation & chamfer loss prepare
            '''
            if data_dict["epoch"] > 20: 
                intr_mat = data_dict["intrinsics"].detach()  # [3,3]
                transform = data_dict["transform"].detach()  # [4,4]
                R_gt = transform[:3,0:3]     # [3,3]
                T_gt = transform[:3,3:4].t() # [1,3] 
                kpts_3d_scores = output_dict["coarse_kpts"]
                kpts_3d_mask = (kpts_3d_scores[:,0] >= 0.40) # 0.35

                # exception process
                num_filter = torch.sum(kpts_3d_mask)
                if num_filter >= 50:
                    kpts_3d_pts = output_dict["kpts_3d_pts"][kpts_3d_mask,:] # [n,3]
                    kpts_3d_pix = output_dict["kpts_3d_pix"][kpts_3d_mask,:] # [n,2]
                    kpts_3d_fea = output_dict["kpts_3d_fea"][kpts_3d_mask,:] # [n,c]
                    kpts_2d_pts = output_dict["kpts_2d_pts"] # [m,3]
                    kpts_2d_pix = output_dict["kpts_2d_pix"] # [m,2]
                    kpts_2d_fea = output_dict["kpts_2d_fea"] # [m,c]

                    m = kpts_2d_pix.size(0)
                    tmp_vector = torch.ones((m,3)).cuda()
                    # tmp_vector[:,0] = kpts_2d_pix[:,1]
                    # tmp_vector[:,1] = kpts_2d_pix[:,0]     # [m,3]
                    inv_intr_mat = torch.inverse(intr_mat)   # [3,3]
                    kpts_2d_bea = torch.matmul(inv_intr_mat, tmp_vector.t()).t() # [m,3]

                    output_dict["kpts_3d_pts"] = kpts_3d_pts
                    output_dict["kpts_3d_pix"] = kpts_3d_pix

                    # keypoints down-sample
                    n = kpts_3d_pts.size(0)
                    if n >= 500:
                        kpts_3d_pts = kpts_3d_pts[::4,:] # ::4 or 5
                        kpts_3d_pix = kpts_3d_pix[::4,:]
                        kpts_3d_fea = kpts_3d_fea[::4,:]
                        output_dict["kpts_3d_pts"] = kpts_3d_pts
                        output_dict["kpts_3d_pix"] = kpts_3d_pix
                    m = kpts_2d_pts.size(0)
                    if m >= 500:
                        kpts_2d_pts = kpts_2d_pts[::2,:]
                        kpts_2d_pix = kpts_2d_pix[::2,:]
                        kpts_2d_fea = kpts_2d_fea[::2,:]
                        kpts_2d_bea = kpts_2d_bea[::2,:]
                        output_dict["kpts_2d_pts"] = kpts_2d_pts
                        output_dict["kpts_2d_pix"] = kpts_2d_pix
                
                    # channel = 131
                    xyz_3d = kpts_3d_pts.t().unsqueeze(0) # [n,3] => [1,3,n]
                    xyz_2d = kpts_2d_bea.t().unsqueeze(0) # [m,3] => [1,3,m]
                    fea_3d = torch.concatenate((kpts_3d_pts, kpts_3d_fea), dim=1)
                    fea_2d = torch.concatenate((kpts_2d_bea, kpts_2d_fea), dim=1)
                    fea_3d = fea_3d.t().unsqueeze(0) # [n,3] => [1,3,n]
                    fea_2d = fea_2d.t().unsqueeze(0) # [m,3] => [1,3,m]
                    fea_2d = self.kpts_2d_pre(fea_2d)
                    fea_3d = self.kpts_3d_pre(fea_3d)
                    fea_2d, _ = self.kpts_net_2d(fea_2d, xyz_2d) # [1,c,m]
                    fea_3d, _ = self.kpts_net_3d(fea_3d, xyz_3d) # [1,c,n]

                    mean_kpts_2d_fea = torch.mean(fea_2d, dim=2) # [1,c]
                    mean_kpts_3d_fea = torch.mean(fea_3d, dim=2) # [1,c]
                    pose_fea = torch.concatenate((mean_kpts_2d_fea, mean_kpts_3d_fea), dim=1) # [1,2c]
                    pose = self.so_pred_net(pose_fea[0]) # [6]
                    R_pd = axis_angle_to_rotation_matrix(pose[0:3]*0.1) # [3,3]
                    T_pd = pose[3:6].unsqueeze(0)                       # [1,3]
                    dR   = rotation_matrix_to_axis_angle(torch.matmul(R_pd.t(), R_gt))
                    dT   = T_pd - T_gt 
                    output_dict["dR"] = dR
                    output_dict["dT"] = dT

                    # projection check --- ok using R_gt and T_gt
                    xyz_3d_tf  = torch.matmul(R_pd, xyz_3d[0])  # [3,n]
                    xyz_3d_tf  = xyz_3d_tf.t() + T_pd           # [n,3]
                    pix_render = render(xyz_3d_tf, intr_mat)
                    output_dict["kpts_3d_pix_render"] = pix_render
                        
                    '''
                        visulization analysis --- ok 
                            kpts_3d_pix and kpts_2d_pix computation --- ok
                            kpts_3d_pts and kpts_2d_pts computation --- ok
                    '''
                    is_need_vis_2d = False
                    if is_need_vis_2d:
                        import cv2 as cv
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
                        # projection check --- ok
                        pix_render = pix_render.detach().cpu().numpy()
                        for i in range(pix_render.shape[0]):
                            x = pix_render[i,1]
                            y = pix_render[i,0]
                            cv.circle(rgb_image, (int(x), int(y)), radius=4, color=(255,255,0), thickness=2)

                        cv.imshow("rgb_image", rgb_image)
                        cv.waitKey(0)
                        print("rgb_image: ", rgb_image.shape)
                        # assert 1==-1

        # 6. Select topk nearest node correspondences
        if not self.training:
            output_dict = self.post_process_generate_corres(
                img_feats_c.detach(), pcd_feats_c.detach(), img_feats_f.detach(), pcd_feats_f.detach(), output_dict)
        # make correspondences differential
        # if True:
        #     output_dict = self.post_process_generate_corres(
        #         img_feats_c, pcd_feats_c, img_feats_f, pcd_feats_f, output_dict)

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline(cfg)
    return model

