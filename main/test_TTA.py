import torch
import copy
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    eval_result = {}
    cur_sample_idx = 0
    kypt_list = []
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # do_flip
        if True:
            inputs['img'] = inputs['img'].flip(3) # dict_keys(['img']), torch.Size([32, 3, 256, 256])
            targets['joints_img'][:,:,0] = (256. - targets['joints_img'][:,:,0]*256. - 1.)/256.
  
            with torch.no_grad():
                flip_out = tester.model(inputs, targets, meta_info, 'test')
                flip_out['joints_img'][:,:,0] = (256. - flip_out['joints_img'][:,:,0]*256. - 1.)/256.
                
                for idx in range(flip_out['mesh_coord_cam'].shape[0]):    
                    flip_out['mesh_coord_cam'][idx] -= flip_out['joints_coord_cam'][idx][0,None,:].clone()
                    flip_out['mesh_coord_cam'][idx][:,0] *= -1
                    flip_out['mesh_coord_cam'][idx][:,2] *= -1
                    out['mesh_coord_cam'][idx] -= out['joints_coord_cam'][idx][0,None,:].clone()
                
                for idx in range(flip_out['joints_coord_cam'].shape[0]):    
                    flip_out['joints_coord_cam'][idx] -= flip_out['joints_coord_cam'][idx][0,None,:].clone()
                    flip_out['joints_coord_cam'][idx][:,0] *= -1
                    flip_out['joints_coord_cam'][idx][:,2] *= -1
                    out['joints_coord_cam'][idx] -= out['joints_coord_cam'][idx][0,None,:].clone()
        
        # print('*'*50)
        # print(out['joints_coord_cam'])
        # print(flip_out['joints_coord_cam'])
        # exit()        
        for key in out.keys():
        #     print('*'*50)
        #     print(out[key])
            ratio = 0.5
            out[key] = out[key]*ratio+flip_out[key]*(1.-ratio)

        #     print(out[key])
        
        # exit()
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        
        kypt_list.extend(out['joints_img'].tolist()) # 2D joints

        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        
        # evaluate
        tester._evaluate(out, cur_sample_idx)
        cur_sample_idx += len(out)
    
    # 保存2D关键点
    import json
    with open('kypt_eval.json','w',encoding='utf8') as f:
        f.write(json.dumps(kypt_list,ensure_ascii=False))

    tester._print_eval_result(args.test_epoch)

if __name__ == "__main__":
    main()