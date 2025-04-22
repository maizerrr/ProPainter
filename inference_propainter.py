# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from diffusers import StableDiffusionInpaintPipeline

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def video_cutter(frames, threshold=0.3):
    """Cuts the video into clips based on the threshold.
    """
    cuts = []
    for i in range(1, frames.size(1)):
        diff = torch.mean(torch.abs(frames[:,i] - frames[:,i-1]), dim=(0,1,2,3)).item()
        if diff > threshold:
            cuts.append(i)
    return cuts


def inference(
        frames, 
        flow_masks, 
        masks_dilated, 
        fix_flow_complete, 
        model, 
        vipdiff_pipe, 
        size,
        device):
    video_length = frames.size(1)
    w, h = size
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640: 
            short_clip_len = 120
        elif frames.size(-1) <= 720: 
            short_clip_len = 80
        elif frames.size(-1) <= 1280:
            short_clip_len = 40
        else:
            short_clip_len = 20
        
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in tqdm(range(0, video_length, short_clip_len), desc="Computing ground truth optical flow:"):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()


        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        
        # ---- complete flow ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in tqdm(range(0, flow_length, args.subvideo_length), desc="Computing complete optical flow:"):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            

        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in tqdm(range(0, video_length, subvideo_length_img_prop), desc="Pixel propagation (pre-processing):"):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            
    
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1
    
    # ---- feature propagation + transformer ----
    if args.mode == 'video_inpainting':
        for f in tqdm(range(0, video_length, neighbor_stride), "Inpainting (ProPainter):"):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()
        comp_frames = np.stack(comp_frames, axis=0)
    # ---- LDM inpainting ----
    elif args.mode == 'video_inpainting_vipdiff':
        comp_frames = None
        b, t, _, h, w = updated_frames.size()
        if b != 1:
            print(f"Warning! VipDiff does not support batch inference, will process {b} batches one by one.")
        if t != video_length:
            raise ValueError(f"Error! The length of updated frames {t} does not match the video length {video_length}.")
        pbar = tqdm(total=b * t, desc="Inpainting (VipDiff):")
        for curr_b in range(b):
            to_inpaint_frames = list(torch.unbind(updated_frames[curr_b:curr_b+1,:,:,:,:], dim=1)) # expected shape (T, 1, 3, H, W)
            to_inpaint_masks = list(torch.unbind(updated_masks[curr_b:curr_b+1,:,:,:,:], dim=1)) # expected shape (T, 1, 1, H, W)
            for f in range(video_length):
                to_inpaint_mask = to_inpaint_masks[f]
                if torch.all(to_inpaint_mask == 0):
                    # no inpainting needed
                    pbar.update(1)
                    continue
                with torch.no_grad():
                    # 1. inpaint first frame with LDM
                    to_inpaint_frame = to_inpaint_frames[f]
                    # inpaint_prompt = "" # dummy prompt
                    # pred_img = vipdiff_pipe(
                    #     prompt=inpaint_prompt, 
                    #     image=to_inpaint_frame, 
                    #     mask_image=to_inpaint_mask, 
                    #     height=h, width=w
                    # ).images[0]

                    # pred with cv2
                    to_inpaint_frame_np = to_inpaint_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 3)
                    to_inpaint_mask_np = to_inpaint_mask.squeeze(0).squeeze(0).cpu().numpy()          # Shape: (H, W)
                    to_inpaint_frame_np = ((to_inpaint_frame_np + 1.0) * 127.5).astype(np.uint8)
                    to_inpaint_mask_np = ((to_inpaint_mask_np > 0).astype(np.uint8) * 255)
                    pred_img = cv2.inpaint(to_inpaint_frame_np, to_inpaint_mask_np, 3, cv2.INPAINT_TELEA)
                    
                    pred_img = torch.from_numpy(np.array(pred_img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
                    to_inpaint_frames[f] = pred_img
                    to_inpaint_masks[f] = torch.zeros_like(to_inpaint_mask).to(device)
                    torch.cuda.empty_cache()
                    # 2. update following frames with pixel propagation
                    if f == video_length - 1:
                        pbar.update(1)
                        continue
                    elif video_length > subvideo_length_img_prop:
                        for _f in range(f + 1, video_length, subvideo_length_img_prop):
                            s_f = max(0, _f - pad_len)
                            e_f = min(video_length, _f + subvideo_length_img_prop + pad_len)
                            pad_len_s = max(0, _f) - s_f
                            pad_len_e = e_f - min(video_length, _f + subvideo_length_img_prop)
                            pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                            to_prop_frame_tensor = torch.stack(to_inpaint_frames[s_f:e_f], dim=1).to(device)
                            to_prop_mask_tensor = torch.stack(to_inpaint_masks[s_f:e_f], dim=1).to(device)
                            prop_imgs_sub, updated_local_masks_sub = model.img_propagation(to_prop_frame_tensor, 
                                                                            pred_flows_bi_sub, 
                                                                            to_prop_mask_tensor, 
                                                                            'nearest')
                            updated_frames_sub = prop_imgs_sub.view(b, -1, 3, h, w) * to_prop_mask_tensor + to_prop_frame_tensor * (1 - to_prop_mask_tensor)
                            updated_masks_sub = updated_local_masks_sub.view(b, -1, 1, h, w)
                            to_inpaint_frames[_f:e_f-pad_len_e] = updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e].unbind(dim=1)
                            to_inpaint_masks[_f:e_f-pad_len_e] = updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e].unbind(dim=1)
                            torch.cuda.empty_cache()
                    else:
                        pred_flows_bi_sub = (pred_flows_bi[0][:, f+1:], pred_flows_bi[1][:, f+1:])
                        to_prop_frame_tensor = torch.stack(to_inpaint_frames[f+1:], dim=1).to(device)
                        to_prop_mask_tensor = torch.stack(to_inpaint_masks[f+1:], dim=1).to(device)
                        prop_imgs_sub, updated_local_masks_sub = model.img_propagation(to_prop_frame_tensor, 
                                                                            pred_flows_bi_sub, 
                                                                            to_prop_mask_tensor, 
                                                                            'nearest')
                        updated_frames_sub = prop_imgs_sub.view(b, -1, 3, h, w) * to_prop_mask_tensor + to_prop_frame_tensor * (1 - to_prop_mask_tensor)
                        updated_masks_sub = updated_local_masks_sub.view(b, -1, 1, h, w)
                        to_inpaint_frames[f+1:] = updated_frames_sub.unbind(dim=1)
                        to_inpaint_masks[f+1:] = updated_masks_sub.unbind(dim=1)
                        torch.cuda.empty_cache()
                pbar.update(1)
            inpainted_frames = torch.stack(to_inpaint_frames, dim=1).squeeze(0).cpu().permute(0, 2, 3, 1).numpy()
            inpainted_frames = (inpainted_frames + 1.0) * 255.0 / 2.0
            inpainted_frames = inpainted_frames.astype(np.uint8)
            comp_frames = np.concatenate((comp_frames, inpainted_frames), axis=0) if comp_frames is not None else inpainted_frames
    # ---- preview propagation result ----
    elif args.mode == 'video_propagation':
        comp_frames = updated_frames.squeeze(0).cpu().permute(0, 2, 3, 1).numpy()
        comp_frames = (comp_frames + 1.0) * 255.0 / 2.0
        comp_frames = comp_frames.astype(np.uint8)
    else:
        raise NotImplementedError(f"Mode {args.mode} is not supported.")
    
    return comp_frames

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/video', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/video_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_inpainting_lama', 'video_inpainting_vipdiff', 'video_propagation', 'video_outpainting'], help="Modes: video_inpainting / video_inpainting_lama / video_inpainting_vipdiff / video_propagation / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    frames, fps, size, video_name = read_frame_from_videos(args.video)
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)
    
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if args.mode == 'video_inpainting' or args.mode == 'video_inpainting_lama' or args.mode == 'video_inpainting_vipdiff' or args.mode == 'video_propagation':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size, 
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError
    
    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1-alpha)*img + alpha*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    
    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()


    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    ##############################################
    # set up VipDiff pipeline
    ##############################################
    vipdiff_pipe = None
    if args.mode == 'video_inpainting_vipdiff':
        vipdiff_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            variant="fp16",
            torch_dtype=torch.float16
        )
        if device == "cpu":
            # cpu can only use fp32
            for param in vipdiff_pipe.parameters():
                param.data = param.data.float()
        vipdiff_pipe = vipdiff_pipe.to(device)
        vipdiff_pipe.set_progress_bar_config(disable=True)

    video_length = frames.size(1)

    ##############################################
    # ProPainter inference
    ##############################################
    comp_frames = None
    cuts = video_cutter(frames)
    print(f"Video cuts: {len(cuts)}")
    if len(cuts) == 0:
        comp_frames = inference(frames, flow_masks, masks_dilated, fix_flow_complete, model, vipdiff_pipe, size, device)
    else:
        start = 0
        cuts.append(video_length)
        for cut in cuts:
            comp_frames_sub = inference(frames[:,start:cut], flow_masks[:,start:cut], masks_dilated[:,start:cut], 
                                        fix_flow_complete, model, vipdiff_pipe, size, device)
            comp_frames = comp_frames_sub if comp_frames is None else np.concatenate((comp_frames, comp_frames_sub), axis=0)
            start = cut

    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    

    # if args.mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
    #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
    
    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
    
    print(f'\nAll results are saved in {save_root}')
    
    torch.cuda.empty_cache()