import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.backbone import FPN
from nets.transformer import Transformer, Mlp
from nets.CAT import CrossAttention
from nets.KGC import KnowledgeGuidedModule
from nets.regressor import Regressor
from utils.mano import MANO
from config import cfg
import math


class Model(nn.Module):
    def __init__(self, backbone, KGC, CAT, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.KGC = KGC
        self.CAT = CAT
        self.regressor = regressor

    def forward(self, inputs, targets, meta_info, mode):
        feats = self.backbone(inputs['img']) # batch_size*256*32*32
        kypt_feats = inputs['joints_img'] # batch_size*21*2

        # KGC module
        kypt_feats = self.KGC(kypt_feats)
        kypt_feats = kypt_feats.view(feats.shape[0],-1,feats.shape[2],feats.shape[3])
        
        # CAT module
        feats = self.CAT(feats, kypt_feats)
       
        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)
       
        if mode == 'train':
            # loss functions
            loss = {}
            loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
            loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
            loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], inputs['joints_img'])
            return loss

        else:
            # test output
            out = {}
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            out['joints_img'] = preds_joints_img[0]
            
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    KGC = KnowledgeGuidedModule(2,1024)
    CAT = CrossAttention(256, 21)
    regressor = Regressor()
    
    if mode == 'train':
        KGC.apply(init_weights)
        CAT.apply(init_weights)
        regressor.apply(init_weights)
        
    model = Model(backbone, KGC, CAT, regressor)
    
    return model