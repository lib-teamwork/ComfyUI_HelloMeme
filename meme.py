import os
import os.path as osp
import random

import torch
import numpy as np
import cv2
import sys


from PIL import Image
import subprocess

import importlib.metadata

cur_dir = osp.dirname(osp.abspath(__file__))

installed_packages = [package.name for package in importlib.metadata.distributions()]

REQUIRED = {
  'diffusers', 'transformers', 'einops', 'opencv-python', 'tqdm', 'pillow', 'onnxruntime-gpu', 'onnx', 'safetensors', 'accelerate', 'peft'
}

missing = [name for name in REQUIRED if name not in installed_packages]
print("missing pkgs", missing)

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

from .hellomeme.utils import (face_params_to_tensor,
                              gen_control_heatmaps,
                              get_drive_params,
                              crop_and_resize,
                              get_face_params,
                              load_data_list,
                              load_unet_from_safetensors
                              )

from .hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo
from .hellomeme import HMImagePipeline, HMVideoPipeline, HMVideoSimplePipeline
from transformers import CLIPVisionModelWithProjection

class HMImagePipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files = sorted(load_data_list(osp.join(cur_dir, '../../models/checkpoints'), '.pt;.pth;.ckpt;.safetensors'))
        lora_files = sorted(load_data_list(osp.join(cur_dir, '../../models/loras'), '.safetensors'))

        return {
            "optional": {
                "checkpoint_path": (['SD1.5'] + checkpoint_files, ),
                "lora_path": (['None'] + lora_files, ),
                "gpu_id": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("HMIMAGEPIPELINE", )
    RETURN_NAMES = ("hm_image_pipeline", )
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"
    def load_pipeline(self, checkpoint_path=None, lora_path=None, gpu_id=0):
        dtype = torch.float16
        if gpu_id >= 0:
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            device = torch.device("cpu")
        pipeline = HMImagePipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(dtype=dtype, device=device)
        pipeline.caryomitosis()

        ### load customized checkpoint or lora here:
        ## checkpoints

        if checkpoint_path and osp.isfile(checkpoint_path):
            if checkpoint_path.endswith('.safetensors'):
                state_dict = load_unet_from_safetensors(checkpoint_path, pipeline.unet_ref.config)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            elif osp.splitext(checkpoint_path)[-1] in ['.pt', '.pth', '.ckpt']:
                state_dict = torch.load(checkpoint_path)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            else:
                print("Invalid checkpoint path", checkpoint_path)

        ### lora
        if lora_path and osp.isfile(lora_path):
            pipeline.load_lora_weights(osp.dirname(lora_path), weight_name=osp.basename(lora_path), adapter_name="lora")

        pipeline.insert_hm_modules(dtype=dtype, device=device)
        
        return (pipeline, )

class HMVideoPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files = sorted(load_data_list(osp.join(cur_dir, '../../models/checkpoints'), '.pt;.pth;.ckpt;.safetensors'))
        lora_files = sorted(load_data_list(osp.join(cur_dir, '../../models/loras'), '.safetensors'))

        return {
            "optional": {
                "checkpoint_path": (['SD1.5'] + checkpoint_files, ),
                "lora_path": (['None'] + lora_files, ),
                "patch_frames": ([12, 16], ),
                "gpu_id": ("INT", {"default": 0}, ),
            }
        }

    RETURN_TYPES = ("HMVIDEOPIPELINE",)
    RETURN_NAMES = ("hm_video_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"

    def load_pipeline(self, checkpoint_path=None, lora_path=None, patch_frames=12, gpu_id=0):
        dtype = torch.float16
        if gpu_id >= 0:
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            device = torch.device("cpu")
        pipeline = HMVideoPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(dtype=dtype,
                                                                                                     device=device)
        pipeline.caryomitosis(patch_frames=patch_frames)

        ### load customized checkpoint or lora here:
        ## checkpoints

        if checkpoint_path and osp.isfile(checkpoint_path):
            if checkpoint_path.endswith('.safetensors'):
                state_dict = load_unet_from_safetensors(checkpoint_path, pipeline.unet_ref.config)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            elif osp.splitext(checkpoint_path)[-1] in ['.pt', '.pth', '.ckpt']:
                state_dict = torch.load(checkpoint_path)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            else:
                print("Invalid checkpoint path", checkpoint_path)

        ### lora
        if lora_path and osp.isfile(lora_path):
            pipeline.load_lora_weights(osp.dirname(lora_path), weight_name=osp.basename(lora_path), adapter_name="lora")

        pipeline.insert_hm_modules(dtype=dtype, device=device)

        return (pipeline,)

class HMVideoSimplePipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files = sorted(load_data_list(osp.join(cur_dir, '../../models/checkpoints'), '.pt;.pth;.ckpt;.safetensors'))
        lora_files = sorted(load_data_list(osp.join(cur_dir, '../../models/loras'), '.safetensors'))

        return {
            "optional": {
                "checkpoint_path": (['SD1.5'] + checkpoint_files, ),
                "lora_path": (['None'] + lora_files, ),
                "gpu_id": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("HMVIDEOPIPELINE",)
    RETURN_NAMES = ("hm_video_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"

    def load_pipeline(self, checkpoint_path=None, lora_path=None, gpu_id=0):
        dtype = torch.float16
        if gpu_id >= 0:
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            device = torch.device("cpu")
        pipeline = HMVideoSimplePipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(dtype=dtype,
                                                                                                     device=device)
        pipeline.caryomitosis()

        ### load customized checkpoint or lora here:
        ## checkpoints

        if checkpoint_path and osp.isfile(checkpoint_path):
            if checkpoint_path.endswith('.safetensors'):
                state_dict = load_unet_from_safetensors(checkpoint_path, pipeline.unet_ref.config)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            elif osp.splitext(checkpoint_path)[-1] in ['.pt', '.pth', '.ckpt']:
                state_dict = torch.load(checkpoint_path)
                pipeline.unet.load_state_dict(state_dict, strict=False)
            else:
                print("Invalid checkpoint path", checkpoint_path)

        ### lora
        if lora_path and osp.isfile(lora_path):
            pipeline.load_lora_weights(osp.dirname(lora_path), weight_name=osp.basename(lora_path), adapter_name="lora")

        pipeline.insert_hm_modules(dtype=dtype, device=device)

        return (pipeline,)

class HMFaceToolkitsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_id": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("FACE_TOOLKITS",)
    RETURN_NAMES = ("face_toolkits",)
    FUNCTION = "load_face_toolkits"
    CATEGORY = "hellomeme"
    def load_face_toolkits(self, gpu_id):
        dtype = torch.float16
        device = torch.device(f'cuda:{gpu_id}')

        face_aligner = HelloCameraDemo(face_alignment_module=HelloFaceAlignment(gpu_id=gpu_id), reset=False)
        harkit_bs = HelloARKitBSPred(gpu_id=gpu_id)
        h3dmm = Hello3DMMPred(gpu_id=gpu_id)
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'h94/IP-Adapter',subfolder='models/image_encoder').to(dtype=dtype, device=device)
        return (dict(face_aligner=face_aligner, harkit_bs=harkit_bs, h3dmm=h3dmm, image_encoder=clip_image_encoder), )

class GetReferenceImageRT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_toolkits": ("FACE_TOOLKITS",),
            }
        }

    RETURN_TYPES = ("REFRT",)
    RETURN_NAMES = ("ref_rt",)
    FUNCTION = "get_reference_image_rt"
    CATEGORY = "hellomeme"

    def get_reference_image_rt(self, image, face_toolkits):
        image_np = cv2.cvtColor((image[0] * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (512, 512))
        # print(image_np.shape)
        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        assert len(faces) > 0
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        ref_rot, ref_trans = face_toolkits['h3dmm'].forward_params(image_np, ref_landmark)
        return (dict(rot=ref_rot, trans=ref_trans), )

class CropReferenceImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_toolkits": ("FACE_TOOLKITS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_reference_image"
    CATEGORY = "hellomeme"

    def crop_reference_image(self, image, face_toolkits):
        image_np = cv2.cvtColor((image[0] * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (512, 512))
        # print(image_np.shape)
        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        assert len(faces) > 0
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        new_image = crop_and_resize(image_np[np.newaxis, :,:,:], ref_landmark[np.newaxis, :,:], 512, crop=True)[0]
        new_image = cv2.cvtColor(new_image[0], cv2.COLOR_RGB2BGR)
        return (torch.from_numpy(new_image[np.newaxis, :,:,:]).float() / 255., )

class GetImageDriveParams:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "image": ("IMAGE",),
                "ref_rt": ("REFRT",),
            }
        }

    RETURN_TYPES = ("DRIVE_PARAMS",)
    RETURN_NAMES = ("drive_params",)
    FUNCTION = "get_face_params"
    CATEGORY = "hellomeme"
    def get_face_params(self, face_toolkits, image, ref_rt):
        dtype = torch.float16

        image_np = cv2.cvtColor((image[0] * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        assert len(faces) > 0

        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        drive_landmark = face['pre_kpt_222']

        drive_face_parts, drive_coeff, drive_rot, drive_trans = get_face_params(face_toolkits['h3dmm'],
                                                                                face_toolkits['harkit_bs'],
                                                                                [image_np],
                                                                                [drive_landmark],
                                                                                save_size=(512, 512),
                                                                                align=False)

        face_parts_embedding = face_params_to_tensor(face_toolkits['image_encoder'], drive_face_parts)

        drive_params = dict(
            face_parts=face_parts_embedding.unsqueeze(0).to(dtype=dtype, device='cpu'),
            drive_coeff=drive_coeff.unsqueeze(0).to(dtype=dtype, device='cpu'),
            drive_rot=drive_rot,
            drive_trans=drive_trans,
        )
        return (drive_params, )

class HMPipelineImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hm_image_pipeline": ("HMIMAGEPIPELINE",),
                "ref_image": ("IMAGE",),
                "ref_rt": ("REFRT",),
                "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "drive_params": ("DRIVE_PARAMS",),
                "prompt": ("STRING", {"default": '(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture'}),
                "negative_prompt": ("STRING", {"default": ''}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 100000}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self, hm_image_pipeline, ref_image, ref_rt, trans_ratio, drive_params,  prompt, negative_prompt, steps=25, seed=-1, guidance_scale=2.0):
        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)

        drive_rot = drive_params['drive_rot']
        drive_trans = drive_params['drive_trans']

        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_rt['trans'], 512, trans_ratio)
        drive_params['condition'] = condition.unsqueeze(0).to(dtype=torch.float16, device='cpu')

        if seed < 0:
            generator = torch.Generator().manual_seed(random.randint(0, 100000))
        else:
            generator = torch.Generator().manual_seed(seed)

        result_img = hm_image_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            output_type='np'
        )
        return (torch.from_numpy(np.clip(result_img[0], 0, 1)), )

class GetVideoDriveParams:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("DRIVE_PARAMS",)
    RETURN_NAMES = ("drive_params",)
    FUNCTION = "get_face_params"
    CATEGORY = "hellomeme"
    def get_face_params(self, face_toolkits, images):
        dtype = torch.float16

        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]

        face_toolkits['face_aligner'].reset_track()
        (drive_face_parts, drive_coeff, drive_rot, drive_trans) = get_drive_params(face_toolkits['face_aligner'],
                                                                                   face_toolkits['h3dmm'],
                                                                                   face_toolkits['harkit_bs'],
                                                                                   frame_list=frame_list,
                                                                                   save_size=512,
                                                                                   align=True)
        face_toolkits['face_aligner'].reset_track()
        face_parts_embedding = face_params_to_tensor(face_toolkits['image_encoder'], drive_face_parts)

        drive_params = dict(
            face_parts=face_parts_embedding.unsqueeze(0).to(dtype=dtype, device='cpu'),
            drive_coeff=drive_coeff.unsqueeze(0).to(dtype=dtype, device='cpu'),
            drive_rot=drive_rot,
            drive_trans=drive_trans,
        )
        return (drive_params, )

class HMPipelineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "hm_video_pipeline": ("HMVIDEOPIPELINE",),
                        "ref_image": ("IMAGE",),
                        "ref_rt": ("REFRT",),
                        "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "drive_params": ("DRIVE_PARAMS",),
                        "prompt": ("STRING", {"default": '(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture'}),
                        "negative_prompt": ("STRING", {"default": ''}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                        "seed": ("INT", {"default": -1, "min": -1, "max": 100000}),
                        "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self, hm_video_pipeline, ref_image, ref_rt, trans_ratio, drive_params,  prompt, negative_prompt, steps=25, seed=-1, guidance_scale=2.0):
        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)
        if seed < 0:
            generator = torch.Generator().manual_seed(random.randint(0, 100000))
        else:
            generator = torch.Generator().manual_seed(seed)

        drive_rot = drive_params['drive_rot']
        drive_trans = drive_params['drive_trans']

        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_rt['trans'], 512, trans_ratio)
        drive_params['condition'] = condition.unsqueeze(0).to(dtype=torch.float16, device='cpu')

        res_frames = hm_video_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            output_type='np'
        )
        res_frames = [np.clip(x[0], 0, 1) for x in res_frames]
        return (torch.from_numpy(np.array(res_frames)), )

NODE_CLASS_MAPPINGS = {
    "HMImagePipelineLoader": HMImagePipelineLoader,
    "HMVideoPipelineLoader": HMVideoPipelineLoader,
    "HMVideoSimplePipelineLoader": HMVideoSimplePipelineLoader,
    "HMFaceToolkitsLoader": HMFaceToolkitsLoader,
    "GetReferenceImageRT": GetReferenceImageRT,
    "GetImageDriveParams": GetImageDriveParams,
    "GetVideoDriveParams": GetVideoDriveParams,
    "HMPipelineImage": HMPipelineImage,
    "HMPipelineVideo": HMPipelineVideo,
    "CropReferenceImage": CropReferenceImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HMImagePipelineLoader": "Load HelloMemeImage Pipeline",
    "HMVideoPipelineLoader": "Load HelloMemeVideo Pipeline",
    "HMVideoSimplePipelineLoader": "Load HelloMemeVideoSimple Pipeline",
    "HMFaceToolkitsLoader": "Load Face Toolkits",
    "GetReferenceImageT": "Get Reference Image Translation",
    "GetImageDriveParams": "Get Drive Image Parameters",
    "GetVideoDriveParams": "Get Drive Video Parameters",
    "HMPipelineImage": "HelloMeme Image Pipeline",
    "HMPipelineVideo": "HelloMeme Video Pipeline",
    "CropReferenceImage": "Crop Reference Image",
}
