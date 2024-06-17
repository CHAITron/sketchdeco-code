import torch
import os
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from pytorch_lightning import seed_everything
import cv2
from einops import rearrange, repeat
from PIL import Image
import numpy as np
from torch import autocast
import time

class LocalColourise:
    def __init__(self, outdir):
        print("initialize local colourisation components...")
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
        os.makedirs(outdir, exist_ok=True)
        self.outpath = outdir
        self.SCALE = 2.5
        self.BATCHSZ = 1
        self.base_count = len(os.listdir(self.outpath))
        # load model SD2.1
        config = OmegaConf.load("./configs/stable-diffusion/v2-inference.yaml")
        ckpt = "./ckpt/sd-v2-1_512-ema-pruned.ckpt"
        model = self.load_model_from_config(config, ckpt, 'cuda:0')    
        self.model = model.to(self.device)
        self.sampler = DPMSolverSampler(self.model)

    def __call__(self, blip_cls, masks_path, global_results, bg, tau, seed):
        seed_everything(seed)
        print("start local colourisation process...")
        print(f"object: {blip_cls.upper()}")
        prompt = f"hyperrealistic {blip_cls} in photography style"
        data = [self.BATCHSZ * [prompt]]
        
        # get coordinates from all the masks
        print("preprocessing masks...")
        scale_list = []
        center_top_list = []
        center_left_list = []
        cropped_mask_list = []
        cropped_local_list = []
        for mask, local in zip(masks_path, global_results):
            mask_img = cv2.imread(mask, 0)
            scale, center_top, center_left, cropped_mask, cropped_local = self.preprocess_mask(mask_img, local)
            scale_list.append(scale)
            center_top_list.append(center_top)
            center_left_list.append(center_left)
            cropped_mask_list.append(cropped_mask)
            cropped_local_list.append(cropped_local)
        print("DONE")
        
        # get background image
        init_image, target_w, target_h = self.load_img(bg)
        init_image = repeat(init_image.to(self.device), '1 ... -> b ...', b=self.BATCHSZ)
        
        # composite in pixel space
        print("performing composition...")
        w_list = []
        h_list = []
        sm_list = []
        tmp_img = init_image
        for i, mask in enumerate(masks_path):
            # perform composition in pixel space
            composited_img, w, h, sm = self.read_bg(
                scale_list[i],
                center_top_list[i],
                center_left_list[i],
                tmp_img,
                cropped_local_list[i],
                cropped_mask_list[i],
                self.BATCHSZ,
                self.device
            )
            tmp_img = composited_img
            w_list.append(w)
            h_list.append(h)
            sm_list.append(sm)
        print("DONE")
        
        # save composited image
        save_img = Image.fromarray(((composited_img/torch.max(composited_img.max(), abs(composited_img.min())) + 1) * 127.5)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
        save_img.save(os.path.join(self.outpath, f"{self.base_count:05}_{blip_cls}_composited.png"))
        
        # local colourisation
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                for prompts in data:
                    # classifier-free prompt inversion
                    c, uc, inv_emb = self.load_model_and_get_prompt_embedding(self.model, prompts, inv=True)
                    
                    T1 = time.time()
                    
                    # composited image latent
                    print("get composited latent...")
                    composited_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(composited_img)) 
                    print("DONE")
                    
                    # get params
                    param_list = []
                    top_rr_list = []
                    bottom_rr_list = []
                    left_rr_list = []
                    right_rr_list = []
                    for i in range(len(masks_path)):
                        param, top_rr, bottom_rr, left_rr, right_rr = self.get_param(
                            target_h, 
                            target_w, 
                            composited_latent,
                            w_list[i], 
                            h_list[i], 
                            center_top_list[i],
                            center_left_list[i],
                        )
                        param_list.append(param)
                        top_rr_list.append(top_rr)
                        bottom_rr_list.append(bottom_rr)
                        left_rr_list.append(left_rr)
                        right_rr_list.append(right_rr)
                    
                    # get composited latent shape
                    shape = [composited_latent.shape[1], composited_latent.shape[2], composited_latent.shape[3]]
                    
                    # encode composited image
                    print("get composited encodings...")
                    z_enc, _ = self.sampler.sample(
                        steps=20, # dpm steps
                        inv_emb=inv_emb,
                        unconditional_conditioning=uc,
                        conditioning=c,
                        batch_size=self.BATCHSZ,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=self.SCALE,
                        eta=0.0, # ddim eta
                        order=2, # dpm order
                        x_T=composited_latent,
                        DPMencode=True,
                    )
                    print("DONE")
                    
                    composited_orig = z_enc.clone()
                    
                    # add noise in XOR region of cropped mask and rectangular region
                    for i in range(len(masks_path)):
                        z_enc = self.add_noise(
                            z_enc, 
                            param_list[i], 
                            sm_list[i], 
                            top_rr_list[i], 
                            bottom_rr_list[i], 
                            left_rr_list[i], 
                            right_rr_list[i]
                        )
                    
                    composited_noise = z_enc.clone()
                    
                    # zero-padding for composited image encoding
                    mask = torch.zeros_like(z_enc, device=self.device)
                    
                    for i in range(len(masks_path)):
                        mask[:, :, param_list[i][0]:param_list[i][1], param_list[i][2]:param_list[i][3]] = 1
                    
                    # final sampling
                    print("generate final result...")
                    samples, _ = self.sampler.sample(
                        steps=20, # dpm step
                        inv_emb=inv_emb,
                        conditioning=c,
                        batch_size=self.BATCHSZ,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=self.SCALE,
                        unconditional_conditioning=uc,
                        eta=0.0, # ddim eta
                        order=2, # dpm order
                        x_T=[composited_orig, composited_orig.clone(), composited_orig.clone(), composited_noise],
                        width=w_list[0],
                        height=h_list[0],
                        segmentation_map=sm_list[0],
                        param=param_list[0],
                        mask=mask,
                        target_height=target_h, 
                        target_width=target_w,
                        center_row_rm=center_top_list[0],
                        center_col_rm=center_left_list[0],
                        tau=tau,
                        )
                    print("DONE")
                    
                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    T2 = time.time()
                    print('Running Time: %s s' % ((T2 - T1)))
                    
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(self.outpath, f"{self.base_count:05}_{blip_cls}.png"))
                        self.base_count += 1


    def read_bg(self, scale, center_row_from_top, center_col_from_left, init_image, ref_img, seg, batch_size, device):
        save_image = init_image.clone()
        
        target_width = target_height = 512
        
        # read foreground image and its segmentation map
        ref_image, width, height, segmentation_map  =self.load_img(ref_img, scale, seg=seg, target_size=(target_width, target_height))
        ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=batch_size)

        segmentation_map_orig = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
        segmentation_map_save = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 3 ...', b=batch_size)
        segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(device)

        top_rr = int((0.5*(target_height - height))/target_height * init_image.shape[2])  # xx% from the top
        bottom_rr = int((0.5*(target_height + height))/target_height * init_image.shape[2])  
        left_rr = int((0.5*(target_width - width))/target_width * init_image.shape[3])  # xx% from the left
        right_rr = int((0.5*(target_width + width))/target_width * init_image.shape[3]) 

        center_row_rm = int(center_row_from_top * target_height)
        center_col_rm = int(center_col_from_left * target_width)

        step_height2, remainder = divmod(height, 2)
        step_height1 = step_height2 + remainder
        step_width2, remainder = divmod(width, 2)
        step_width1 = step_width2 + remainder
            
        # compositing in pixel space for same-domain composition
        save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] \
                = save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2].clone() \
                * (1 - segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]) \
                + ref_image[:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                * segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]

        # save the mask and the pixel space composited image
        save_mask = torch.zeros_like(init_image) 
        save_mask[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] = 1

        return save_image, width, height, segmentation_map
    
    def load_model_and_get_prompt_embedding(self, model, prompts, inv=False):
        if inv:
            inv_emb = model.get_learned_conditioning(prompts, inv)
            c = uc = inv_emb
        else:
            inv_emb = None
            
        uc = model.get_learned_conditioning(self.BATCHSZ * [""])
        c = model.get_learned_conditioning(prompts)
            
        return c, uc, inv_emb

    def add_noise(self, z_enc, param, sm, top_rr, bottom_rr, left_rr, right_rr):
        z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
            = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
            * sm[:, :, top_rr:bottom_rr, left_rr:right_rr] \
            + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=self.device) \
            * (1 - sm[:, :, top_rr:bottom_rr, left_rr:right_rr])
        
        return z_enc

    @staticmethod
    def load_model_from_config(config, ckpt, gpu, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location=gpu)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        # model.cuda()
        model.eval()
        return model
    
    @staticmethod
    def preprocess_mask(mask, local):
        # Threshold the image to create binary image
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # Find the contours of the white region in the image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        # Calculate the center of the rectangle
        center_x = x + w / 2
        center_y = y + h / 2
        # Calculate the percentage from the top and left
        center_top = round(center_y / 512, 2)
        center_left = round(center_x / 512, 2)

        aspect_ratio = h / w
        
        if aspect_ratio > 1:  
            scale = w * aspect_ratio / 256  
            scale = h / 256
        else:  
            scale = w / 256
            scale = h / (aspect_ratio * 256) 
                
        scale = round(scale, 2)

        # crop the rectangular region out
        cropped_mask = mask[y:y+h, x:x+w]
        # crop local region
        local = np.array(local)
        cropped_local = local[y:y+h, x:x+w]
        
        return scale, center_top, center_left, cropped_mask, cropped_local
    
    @staticmethod
    def load_img(image, SCALE=None, pad=False, seg=[], target_size=None):
        if isinstance(seg, np.ndarray):
            # Load the input image and segmentation map
            image = Image.fromarray(image).convert("RGB")
            seg_map = Image.fromarray(seg).convert("1")

            # Get the width and height of the original image
            w, h = image.size

            # Calculate the aspect ratio of the original image
            aspect_ratio = h / w

            # Determine the new dimensions for resizing the image while maintaining aspect ratio
            if aspect_ratio > 1:
                new_w = int(SCALE * 256 / aspect_ratio)
                new_h = int(SCALE * 256)
            else:
                new_w = int(SCALE * 256)
                new_h = int(SCALE * 256 * aspect_ratio)

            # Resize the image and the segmentation map to the new dimensions
            image_resize = image.resize((new_w, new_h))
            segmentation_map_resize = cv2.resize(np.array(seg_map).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Pad the segmentation map to match the target size
            padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
            start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
            start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
            padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0], start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize

            # Create a new RGB image with the target size and place the resized image in the center
            padded_image = Image.new("RGB", target_size)
            start_x = (target_size[0] - image_resize.width) // 2
            start_y = (target_size[1] - image_resize.height) // 2
            padded_image.paste(image_resize, (start_x, start_y))

            # Update the variable "image" to contain the final padded image
            image = padded_image
        else:
            w, h = image.size        
            w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
            w = h = 512
            image = image.resize((w, h), resample=Image.LANCZOS)
            
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        
        if isinstance(seg, np.ndarray):
            return 2. * image - 1., new_w, new_h, padded_segmentation_map
        
        return 2. * image - 1., w, h
    
    @staticmethod
    def get_param(target_height, target_width, init_latent, w, h, center_top,center_left):
        # ref's location in ref image in the latent space
        top_rr = int((0.5*(target_height - h))/target_height * init_latent.shape[2])  
        bottom_rr = int((0.5*(target_height + h))/target_height * init_latent.shape[2])  
        left_rr = int((0.5*(target_width - w))/target_width * init_latent.shape[3])  
        right_rr = int((0.5*(target_width + w))/target_width * init_latent.shape[3]) 
                                
        new_height = bottom_rr - top_rr
        new_width = right_rr - left_rr
        
        step_height2, remainder = divmod(new_height, 2)
        step_height1 = step_height2 + remainder
        step_width2, remainder = divmod(new_width, 2)
        step_width1 = step_width2 + remainder
        
        center_row_rm = int(center_top * init_latent.shape[2])
        center_col_rm = int(center_left * init_latent.shape[3])
        
        param = [max(0, int(center_row_rm - step_height1)), 
                min(init_latent.shape[2] - 1, int(center_row_rm + step_height2)),
                max(0, int(center_col_rm - step_width1)), 
                min(init_latent.shape[3] - 1, int(center_col_rm + step_width2))]
        
        return param, top_rr, bottom_rr, left_rr, right_rr
    