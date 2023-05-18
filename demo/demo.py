import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import pickle
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import json
import copy
from pycocotools.coco import COCO

sys.path.insert(0, osp.abspath(osp.join('./', 'main')))
sys.path.insert(0, osp.abspath(osp.join('./', 'common')))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.vis import save_obj
from utils.mano import MANO
from utils.vis import render_mesh, vis_mesh, draw_handpose
from utils.transforms import cam2pixel

mano = MANO()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './output/model_dump_HO3D/snapshot_70.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

with open('./data/HO3D/evaluation.txt','r',encoding='utf8') as f:
    file_list = [id.replace('\n','') for id in f.readlines()]
db = COCO('./data/HO3D/annotations/HO3D_evaluation_data.json')
pred_joints_coord_img = json.load(open('./data/HO3D/kypt_eval.json','r',encoding='utf8'))

from tqdm import tqdm
for idx, file_id in tqdm(enumerate(file_list)):
    try:
        ann = db.anns[idx]
        item, img_id = file_id.split('/')
        file_name = db.loadImgs(ann['image_id'])[0]['file_name']
        assert file_name==item+'/rgb/'+img_id+'.png'

        # prepare input image
        transform = transforms.ToTensor()
        img_path = os.path.join('./data/HO3D/evaluation',file_name)
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]

        # prepare bbox
        bbox = ann['bbox'] # xmin, ymin, width, height 
        bbox = process_bbox(bbox, original_img_width, original_img_height, expansion_factor=1.5)

        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        
        joints_img = torch.tensor([pred_joints_coord_img[idx]],device='cuda')
        
        # forward
        inputs = {'img': img, 'joints_img':joints_img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')
        img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        verts_out = out['mesh_coord_cam'][0].cpu().numpy()
        joints_out = out['joints_coord_cam'][0].cpu().numpy()

        # save mesh (obj)
        save_obj(verts_out, mano.face, './output/obj/'+str(idx)+'.obj')

        cam_param = ann['cam_param']
        root_joint_cam = np.array(ann['root_joint_cam'])
        # img = np.ones((640,480,3))
        
        img = cv2.imread(img_path)
        verts_out = verts_out - joints_out[0] + root_joint_cam
        
        handMesh = render_mesh(img, verts_out, mano.face, cam_param)
    
        cv2.imwrite('./output/vis/'+'handMesh_'+str(idx)+'.png',handMesh,[int(cv2.IMWRITE_PNG_COMPRESSION),0])

        # print(idx)
    except:
        pass