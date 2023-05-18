import os
import cv2
import json
import pickle
import numpy as np
import sys
import copy
import random

from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.mano import MANO
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation

mano = MANO()

def load_json(json_path):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def gen_train_json_anno(root_dir,save_path=''):
    txt_list = open('/raid/Dataset/HO3D_v3/train.txt','r',encoding='utf8').readlines()
    txt_list = [item.replace('\n','') for item in txt_list]
    random.shuffle(txt_list)
    img_id=0
    json_dict = {'images':[], 'annotations':[]}
    reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

    for file_path in txt_list:
        item, file_id = file_path.split('/')
        pkl_path = os.path.join(root_dir, item, 'meta', file_id+'.pkl')
        file_name = os.path.join(item, 'rgb', file_id+'.jpg')
        img_path = os.path.join(root_dir, item, 'rgb', file_id+'.jpg')

        with open(pkl_path,'rb') as f:
            pkl_data = pickle.load(f)
            try:
                img = cv2.imread(img_path)
                width, height = img.shape[1],img.shape[0]
                json_dict['images'].append({'id':img_id, 'file_name':file_name,'width':width,'height':height})
                
                handJoints3D = pkl_data['handJoints3D'][reorder_idxs]

                json_dict['annotations'].append({'id':img_id, 'image_id':img_id,'joints_coord_cam':(handJoints3D*np.array([1.,-1.,-1.],dtype=np.float32)).tolist(),\
                'cam_param':{'focal':[float(pkl_data['camMat'][0,0]),float(pkl_data['camMat'][1,1])],
                'princpt':[float(pkl_data['camMat'][0,2]),float(pkl_data['camMat'][1,2])]},\
                'mano_param': {'pose':pkl_data['handPose'].tolist(),'shape':pkl_data['handBeta'].tolist()}
                })
                img_id+=1
            except Exception as e:
                print(pkl_path)
                print(e)
        
    with open(save_path, 'w') as json_file:
        json_file.write(json.dumps(json_dict,ensure_ascii=False))

def gen_eval_json_anno(root_dir,save_path=''):
    txt_list = open('/raid/Dataset/HO3D_v3/evaluation.txt','r',encoding='utf8').readlines()
    txt_list = [item.replace('\n','') for item in txt_list]
    json_dict = {'images':[], 'annotations':[]}

    img_id=0
    for file_path in txt_list:
        item, file_id = file_path.split('/')
        pkl_path = os.path.join(root_dir, item, 'meta', file_id+'.pkl')
        file_name = os.path.join(item, 'rgb', file_id+'.jpg')
        img_path = os.path.join(root_dir, item, 'rgb', file_id+'.jpg')

        with open(pkl_path,'rb') as f:
            pkl_data = pickle.load(f)
            try:
                img = cv2.imread(img_path)
                width, height = img.shape[1], img.shape[0]
                
                bbox = copy.deepcopy(pkl_data['handBoundingBox'])
                bbox[2]-=bbox[0]
                bbox[3]-=bbox[1]

                json_dict['images'].append({'id':img_id, 'file_name':file_name,'width':width,'height':height})
                json_dict['annotations'].append({'id':img_id, 'image_id':img_id,\
                'root_joint_cam':(pkl_data['handJoints3D']*np.array([1.,-1.,-1.],dtype=np.float32)).tolist(),\
                'cam_param':{'focal':[float(pkl_data['camMat'][0,0]),float(pkl_data['camMat'][1,1])],
                'princpt':[float(pkl_data['camMat'][0,2]),float(pkl_data['camMat'][1,2])]},\
                'bbox': bbox
                })
                img_id+=1
            except Exception as e:
                print(pkl_path)
                print(e)
            
    with open(save_path, 'w') as json_file:
        json_file.write(json.dumps(json_dict,ensure_ascii=False))

def gen_test_json_anno(root_dir,save_path=''):
    txt_list = open('/raid/Dataset/HO3D_v3/train.txt','r',encoding='utf8').readlines()
    txt_list = [item.replace('\n','') for item in txt_list][:100]
    img_id=0
    json_dict = {'images':[], 'annotations':[]}
    reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    for file_path in txt_list:
        item, file_id = file_path.split('/')
        pkl_path = os.path.join(root_dir, item, 'meta', file_id+'.pkl')
        file_name = os.path.join(item, 'rgb', file_id+'.jpg')
        img_path = os.path.join(root_dir, item, 'rgb', file_id+'.jpg')

        with open(pkl_path,'rb') as f:
            pkl_data = pickle.load(f)
        
            img = cv2.imread(img_path)
            width, height = img.shape[1],img.shape[0]
            json_dict['images'].append({'id':img_id, 'file_name':file_name,'width':width,'height':height})

            root_joint_cam = pkl_data['handJoints3D'][reorder_idxs][0]
            
            focal = np.array([float(pkl_data['camMat'][0,0]),float(pkl_data['camMat'][1,1])],dtype=np.float32)
            princpt = np.array([float(pkl_data['camMat'][0,2]),float(pkl_data['camMat'][1,2])],dtype=np.float32)

            joints_coord_cam = pkl_data['handJoints3D'][reorder_idxs]*np.array([1.,-1.,-1.],dtype=np.float32)
            joints_coord_img = cam2pixel(joints_coord_cam, focal, princpt)
            
            bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
            bbox = process_bbox(bbox, 640, 480, expansion_factor=1.0)

            json_dict['annotations'].append({'id':img_id, 'image_id':img_id,'root_joint_cam':(root_joint_cam*np.array([1.,-1.,-1.],dtype=np.float32)).tolist(),\
            'cam_param':{'focal':[float(pkl_data['camMat'][0,0]),float(pkl_data['camMat'][1,1])],
            'princpt':[float(pkl_data['camMat'][0,2]),float(pkl_data['camMat'][1,2])]},\
            'bbox':bbox.tolist()
            })

            img_id+=1
           
        
    with open(save_path, 'w') as json_file:
        json_file.write(json.dumps(json_dict,ensure_ascii=False))
if __name__=='__main__':
    '''生成HO3D的标注文件'''
    eval_dir = 'data/HO3Dv3/evaluation'
    train_dir = 'data/HO3Dv3/train'
    gen_test_json_anno(train_dir,'data/HO3Dv3/annotations/HO3D_evaluation_data.json')
    gen_eval_json_anno(eval_dir,'data/HO3Dv3/annotations/HO3D_evaluation_data.json')
    gen_train_json_anno(train_dir,'data/HO3Dv3/annotations/HO3D_train_data.json')



