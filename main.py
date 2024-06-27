import sys
sys.path.append('/ours')
import json
import torch
from dataset import image_title_dataset
import lpips
from net import highway_buildingNet
from CLIP import clip
from guidediffusion.utils import create_model_and_diffusion,model_and_diffusion_defaults
from losses import range_loss, reconstruct_loss, clip_loss,EdgeLoss,extract_line,footprint2, Two_Pass,footprint,get_compactness_cost
from utils.visualization import show_tensor_image, show_editied_masked_image,save_editied_masked_image,show_result,save_pil
from arguments import get_arguments
from torch.utils import data
from augmentations import ImageAugmentations
from torchvision import transforms
from CannyEdgePytorch.net_canny import CannyNet
import os
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
#please see https://github.com/cvg/SOLD2
from sold.model.line_matcher import LineMatcher
import yaml
import cv2

def load_config(config_path):
    """ Load configurations from a given yaml file. """
    # Check file exists
    if not os.path.exists(config_path):
        raise ValueError("[Error] The provided config path is not valid.")

    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

def inference_one(args):
    device =args.device
    print("device",device)
    #diffusion model
    d_model_config =model_and_diffusion_defaults()
    d_model_config.update(
        {
    "attention_resolutions": "32, 16, 8",
    "class_cond": args.model_output_size == 512,
    "diffusion_steps": 1000,
    "rescale_timesteps": True,
    "timestep_respacing": args.timestep_respacing,
    "image_size": args.model_output_size,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": True,
    "use_scale_shift_norm": True,  
    }
    )
    print("load diffusion")
    d_model, diffusion =create_model_and_diffusion(**d_model_config)
    d_model.load_state_dict(
        torch.load("checkpoints/diffusion.pt",map_location='cpu')
    )

    d_model.requires_grad_(False).eval().to(device)
    for name, param in d_model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
    if d_model_config['use_fp16']:
        d_model.convert_to_fp16()
    print("DONE")

    print('atributes')
    attributes_model = highway_buildingNet()
    prompt_h,prompt_b = attributes_model(args.prompt)
    print("load clip")
    clip_model =clip.load("ViT-B/32",device =device,jit=False)[0]
    #fine-tune
    clip_model.load_state_dict(
        torch.load('.CLIP/building/1120_b/clip.pt',map_location='cpu')['model_state_dict']
    )
    clip_model.requires_grad_(False).eval().to(device)

    print('load clip_r')
    clip_model_r = clip.load("ViT-B/32",device=device,jit=False)[0]
    clip_model_r.load_state_dict(
        torch.load('/CLIP/road/1109_r/clip.pt', map_location='cpu')['model_state_dict']
    )
    clip_model_r.requires_grad_(False).eval().to(device)
    print("DNOE")
    
    #load linedetection
    line_config = load_config('/ours/sold/config/export_line_features.yaml')
    line_matcher =LineMatcher(
        line_config['model_cfg'],'/ours/sold/checkpoint/sold2_wireframe.tar', device, line_config['line_detector_cfg'],
        line_config['line_matcher_cfg'],False
    )

    #loss
    image_augmentations =ImageAugmentations(clip_model.visual.input_resolution,args.aug_num)
    clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    edge_net =CannyNet(threshold=200.0,use_cuda=True).eval().to(device)
    #edge_model =EdgeLoss(edge_net).eval().to(device)
    #input part
    text_embed_h = clip_model_r.encode_text(
            clip.tokenize(prompt_h).to(device)
        ).float()
    text_embed_b =clip_model.encode_text(
        clip.tokenize(prompt_b).to(device)
    ).float()
    image_size = (d_model_config["image_size"], d_model_config["image_size"])
    init_image_pil = Image.open(args.init_image).convert("RGB")
    init_image_pil = init_image_pil.resize(image_size, Image.LANCZOS)
    init_image = (
            TF.to_tensor(init_image_pil).to(device).unsqueeze(0).mul(2).sub(1)
        )

    
    init_image_pil_r = Image.open(args.init_image_r).convert("RGB")
    init_image_pil_r = init_image_pil_r.resize(image_size, Image.LANCZOS)
    init_image_r = (
            TF.to_tensor(init_image_pil_r).to(device).unsqueeze(0).mul(2).sub(1)
        )
    
    init_image_pil_b = Image.open(args.init_image_b).convert("RGB")
    init_image_pil_b = init_image_pil_b.resize(image_size, Image.LANCZOS)
    init_image_b = (
            TF.to_tensor(init_image_pil_b).to(device).unsqueeze(0).mul(2).sub(1)
        )
    
    #base
    init_road =Image.open("/ours/road.png").convert("RGB")
    init_road =init_road.resize(image_size, Image.LANCZOS)
    init_road = (
        TF.to_tensor(init_road).to(device).unsqueeze(0).mul(2).sub(1)
    )
    init_background = Image.open("/ours/background.png").convert("RGB")
    init_background =init_background.resize(image_size, Image.LANCZOS)
    init_background = (
        TF.to_tensor(init_background).to(device).unsqueeze(0).mul(2).sub(1)
    )

    init_building = Image.open("/ours/building.png").convert("RGB")
    init_building =init_building.resize(image_size, Image.LANCZOS)
    init_building = (
        TF.to_tensor(init_building).to(device).unsqueeze(0).mul(2).sub(1)
    )


    mask = torch.ones_like(init_image_r, device=device)
    mask_pil = None
    if args.mask is not None:
        mask_pil = Image.open(args.mask).convert("RGB")
        if mask_pil.size != image_size:
            mask_pil = mask_pil.resize(image_size, Image.NEAREST)  # type: ignore
        image_mask_pil_binarized = ((np.array(mask_pil) > 0.5) * 255).astype(np.uint8)
        if args.invert_mask:
            image_mask_pil_binarized = 255 - image_mask_pil_binarized
            mask_pil = TF.to_pil_image(image_mask_pil_binarized)
        mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
        mask = mask[0, ...].unsqueeze(0).unsqueeze(0).to(device)

    def cond_fn_h(x,t,y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            t =(t* (diffusion.num_timesteps /1000)).long()

            out =diffusion.p_mean_variance(d_model,x,t, clip_denoised=False, model_kwargs={'y':y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
            x_in =out['pred_xstart'] * fac + x* (1-fac)
            
            loss = torch.tensor(0)

            if args.clip_guidance_lambda !=0:
                clip_losses = clip_loss(x_in, text_embed_h,mask, clip_model_r,args.batch_size,
                image_augmentations,clip_normalize
                ) * args.clip_guidance_lambda
                loss=loss+clip_losses
                #print("clip_loss",clip_losses)
            if args.reconstruct_lambda !=0:
                r_loss_h =  reconstruct_loss(x_in,init_image_r,mask,args.target_factor) * args.reconstruct_lambda
                loss =loss+r_loss_h
            if args.range_lambda != 0:
                r_loss= range_loss(out['pred_xstart']).sum() *args.range_lambda
                loss =loss+r_loss
            
            if args.lpips_sim_lambda:
                lpips_loss = lpips_model(x_in*(1-mask), init_image_r).sum() *args.lpips_sim_lambda
                loss = loss+ lpips_loss
            print("h",loss)
        return -torch.autograd.grad(loss,x)[0]
        
    def cond_fn_b(x,t,y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            t =(t* (diffusion.num_timesteps /1000)).long()
            #t = unscale_timestep(t)
            out = diffusion.p_mean_variance(d_model,x,t, clip_denoised=False, model_kwargs={'y':y})
            fac =diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
            x_in =out['pred_xstart'] * fac + x* (1-fac)
            
            loss2 = torch.tensor(0)
            if args.clip_guidance_lambda !=0:
                clip_losses = clip_loss(x_in, text_embed_b,mask, clip_model,args.batch_size,
                image_augmentations,clip_normalize
                ) *args.clip_guidance_lambda
                loss2=loss2+clip_losses
                #print("clip_loss",clip_losses)
            if args.reconstruct_lambda !=0:
                r_loss_h =  reconstruct_loss(x_in,init_image_b,mask,args.target_factor) * args.reconstruct_lambda
                loss2 =loss2+r_loss_h
            if args.range_lambda != 0:
                r_loss= range_loss(out['pred_xstart']).sum() *args.range_lambda
                loss2 =loss2+r_loss
            
            if args.lpips_sim_lambda:
                lpips_loss = lpips_model(x_in*(1-mask), init_image_b).sum() *args.lpips_sim_lambda
                loss2 = loss2+ lpips_loss
            
            if args.foorprint_lambda:
                edege_loss = EdgeLoss(x_in,init_image_b,edge_net,mask) * args.foorprint_lambda
                loss2 = loss2+edege_loss
                print(edege_loss)
            if args.compact_lambda:
                compact_loss = get_compactness_cost(x_in*mask) *args.compact_lambda
                loss2 =loss2+compact_loss
        return -torch.autograd.grad(loss2,x)[0]

    @torch.no_grad()
    def postprocess_fn(out,t):
        if mask is not None:
            background_stage_t_h =diffusion.q_sample(init_image_r,t[0])
            background_stage_t_b =diffusion.q_sample(init_image_b,t[0])
            #print(background_stage_t_h)
            #exit()
            image_h = background_stage_t_h[0].add(1).div(2).clamp(0,1)
            image_h = TF.to_pil_image(image_h)
            image_h.save("/ours/sample_r2/{}.png".format(t))
            image_b = background_stage_t_b[0].add(1).div(2).clamp(0,1)
            image_b = TF.to_pil_image(image_b) 
            image_b.save("/sample_b2/{}.png".format(t))
            #background_stage_t =diffusion.q_sample(img_p, t[0])
            background_stage_t_h =torch.tile(
                background_stage_t_h, dims=(args.batch_size,1,1,1))
            background_stage_t_b =torch.tile(
                background_stage_t_b, dims=(args.batch_size,1,1,1))
                
            mask_background_t =diffusion.q_sample(init_background,t[0]) 
            building_background_t = diffusion.q_sample(init_building,t[0])
            line_background_t =diffusion.q_sample(init_road,t[0])
            #road mask
            line_heatmap ,ref_heatmap2= extract_line(out['sample_h'],line_matcher)
            #print(line_heatmap.max())
            #image =TF.to_pil_image(line_heatmap)
            image =Image.fromarray((line_heatmap*255).astype(np.uint8))
            image.save("/ours/lines/{}.png".format(t))
            line_heatmap=torch.from_numpy(line_heatmap).to(device)
            line_heatmap = line_heatmap * mask
            ref_heatmap2 =torch.from_numpy(ref_heatmap2).to(device)
            #building mask
            building_heatmap =footprint(out['sample_b'])
            cv2.imwrite("/ours/footprint/{}.png".format(t),building_heatmap*255)
            building_heatmap = torch.from_numpy(building_heatmap).to(device)
            building_heatmap = torch.tile(
                building_heatmap,dims =(args.batch_size,1,1,1)
            ).float()
            building_heatmap =building_heatmap * mask
            out['sample_h']=(out['sample_h']* (1-line_heatmap)+line_background_t * (line_heatmap))*mask +out['sample_h']*(1-mask)
            out['sample_h']=(mask_background_t * (building_heatmap)+out['sample_h']*(1-building_heatmap))*mask +out['sample_h']*(1-mask)
            out['sample_b'] =(building_background_t *building_heatmap + out['sample_b']* (1-building_heatmap))*mask +out['sample_b']*(1-mask)
            out['sample_b'] =(mask_background_t * line_heatmap+ out['sample_b'] *(1-line_heatmap))*mask +out['sample_b']*(1-mask)

            image2 = out['sample_h'][0].add(1).div(2).clamp(0,1)
            image2 = TF.to_pil_image(image2)
            image2_gray =TF.to_grayscale(image2)
            image2_gray.save("/ours/line/{}.png".format(t))

            image3 =out['sample_b'][0].add(1).div(2).clamp(0,1)
            image3 =TF.to_pil_image(image3)
            image3_gray = TF.to_grayscale(image3)
            image3_gray.save("/ours/buildings/{}.png".format(t))

            #ret =footprint2(out['sample_b'],edge_net)
            #ret =TF.to_pil_image(ret[0])
            #ret.save("/ours/footprint/{}.png".format(t))
            
            #exit()
            out['sample_h'] =out['sample_h'] * mask+background_stage_t_h * (1-mask)
            out['sample_b'] =out['sample_b'] * mask+background_stage_t_b * (1-mask)
        return out
    save_image_interval =diffusion.num_timesteps // 5
    for iteration_number in range(args.iterations_num):
        count=0
        print(f"Start iterations {iteration_number}")
        model_kwargs={} 
        sample_func =(diffusion.p_sample_loop_progressive)
        samples =sample_func(
                    d_model,
                (args.batch_size,3, args.image_size, args.image_size),
                clip_denoised =False,
                model_kwargs =model_kwargs,
                cond_fn_h =cond_fn_h,
                cond_fn_b =cond_fn_b,
                progress =True,
                skip_timesteps =args.skip_timesteps,
                init_image_h =init_image_r,
                init_image_b =init_image_b,
                postprocess_fn = postprocess_fn,
                randomize_class =True
            )
        intermediate_samples = [[] for i in range(args.batch_size)]
        total_steps = diffusion.num_timesteps - args.skip_timesteps - 1
        for j, sample in enumerate(samples):
            should_save_image = j % save_image_interval == 0 or j == total_steps
            if should_save_image:
                for b in range(args.batch_size):
                    
                    pred_img_h = sample['pred_xstart_h'][b]
                    pred_img_b = sample['pred_xstart_b'][b]
                    pred_img_hh= sample['sample_h'][b]
                    pred_img_bb =sample['sample_b'][b]

                    visualization_path_h =f"{iteration_number}_{b}_{count}"
                    if (mask is not None
                    and args.enforce_background
                    and j==total_steps
                    ):
                        pred_img_h = init_image_r[b]* (1-mask[b]) +pred_img_h *mask[b]
                        pred_img_b = init_image_b[b]* (1-mask[b]) +pred_img_b *mask[b]

                    pred_img_h = init_image_r[b]* (1-mask[b]) +pred_img_h *mask[b]
                    pred_img_b = init_image_b[b]* (1-mask[b]) +pred_img_b *mask[b]
                    pred_img_h=pred_img_h.add(1).div(2).clamp(0,1)
                    pred_img_b=pred_img_b.add(1).div(2).clamp(0,1)


                    pred_img_hh = pred_img_hh.add(1).div(2).clamp(0,1)
                    pred_img_bb = pred_img_bb.add(1).div(2).clamp(0,1)
                    
                    pred_img_hh_pil = TF.to_pil_image(pred_img_hh)
                    pred_img_bb_pil = TF.to_pil_image(pred_img_bb)
                    if should_save_image:
                        save_pil(
                            m=mask_pil,
                            gt_a = init_image_pil,
                            r_h = pred_img_hh_pil,
                            r_b = pred_img_bb_pil,
                            p=visualization_path_h,
                            line_matcher =line_matcher
                        )
                        count = count+1

                        
            
if __name__=='__main__':
    args = get_arguments()
    list_image_path = ['' , '']
    list_txt = ['', '']
    with open('/data/our_caption/train.json') as f :
        gj =json.load(f)
    list_image_path =list(gj.keys())
    list_txt = list(gj.values())
    train_dataset=image_title_dataset(list_image_path=list_image_path,list_txt=list_txt)
    train_dataloader =data.DataLoader(train_dataset,batch_size=args.batch_size, num_workers=16,shuffle=False)

    inference_one(args)

                    

                        











