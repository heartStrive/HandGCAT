import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton
from utils.mano import MANO
mano = MANO()

class HO3Dv3(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split if data_split == 'train' else 'evaluation'
        self.root_dir = osp.join('.', 'data', 'HO3Dv3', '')
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0
        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        pred_path = osp.join(self.root_dir,'kypt_eval.json')
        self.pred_joints_coord_img = json.load(open(pred_path,'r',encoding='utf8'))

        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval_result = [[],[]] #[pred_joints_list, pred_verts_list]
        self.joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinly_4')
    
    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_{}_data.json".format(self.data_split)))
        
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, self.data_split, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) # meter
                cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
                bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam, "joints_coord_img": joints_coord_img,
                        "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape}
            else:
                root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
                cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                bbox = np.array(ann['bbox'], dtype=np.float32)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.5)
                
                data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
                        "bbox": bbox, "cam_param": cam_param}

            datalist.append(data)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, do_flip=False)
        img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            joints_img_xy1 = np.concatenate((joints_img[:,:2], np.ones_like(joints_img[:,:1])),1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # normalize to [0,1]
            joints_img[:,0] /= cfg.input_img_shape[1]
            joints_img[:,1] /= cfg.input_img_shape[0]

            ## 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx,None,:] # root-relative
        
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1,0)).transpose(1,0)

            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1,3)
            root_pose = mano_pose[self.root_joint_idx,:]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            inputs = {'img': img, 'joints_img': joints_img}
            targets = {'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
            meta_info = {'root_joint_cam': root_joint_cam}

        else:
            ## 2D joint coordinate
            joints_img = np.array(self.pred_joints_coord_img[idx], dtype=np.float32)
            joints_img = joints_img[self.reorder_idxs]
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img, 'joints_img': joints_img}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam}

        # from utils.vis import draw_handpose
        # joints_img[:,0] *= cfg.input_img_shape[1]
        # joints_img[:,1] *= cfg.input_img_shape[0]
        # canvas = draw_handpose(img, [joints_img])
        # cv2.imwrite('../../output/vis/eval_'+str(idx)+'.png',canvas)

        return inputs, targets, meta_info
                  

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            
            out = outs[n]
            
            verts_out = out['mesh_coord_cam']
            joints_out = out['joints_coord_cam']
            
            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            
            # img_path, img_shape, bbox = annot['img_path'], annot['img_shape'], annot['bbox']
            # cam_param = {k:np.array(v, dtype=np.float32) for k,v in annot['cam_param'].items()}
            # joints_coord_img = cam2pixel(joints_out, cam_param['focal'], cam_param['princpt'])
            # img = load_img(img_path)
            # # print(joints_coord_img)
            # for point in joints_coord_img:
            #     cv2.circle(img,(int(point[0]),int(point[1])),3,(0,0,255))
            # cv2.imwrite('../../output/vis/pred_'+str(n)+'.png',img)

            # convert  to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            joints_out = transform_joint_to_other_db(joints_out, mano.joints_name, self.joints_name)

            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())

    def print_eval_result(self, test_epoch):
        output_json_file = osp.join(cfg.result_dir, 'pred{}.json'.format(test_epoch)) 
        output_zip_file = osp.join(cfg.result_dir, 'pred{}.zip'.format(test_epoch))
        
        with open(output_json_file, 'w') as f:
            json.dump(self.eval_result, f)
        print('Dumped %d joints and %d verts predictions to %s' % (len(self.eval_result[0]), len(self.eval_result[1]), output_json_file))

        cmd = 'zip -j ' + output_zip_file + ' ' + output_json_file
        print(cmd)
        os.system(cmd)
