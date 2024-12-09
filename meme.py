import os.path as osp
import random

import numpy as np
import cv2
import sys

from PIL import Image
import subprocess

import torch
from einops import rearrange

import importlib.metadata
import folder_paths

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

from .hellomeme.utils import (get_drive_expression,
                              get_drive_expression_pd_fgc,
                              gen_control_heatmaps,
                              get_drive_pose,
                              crop_and_resize,
                              det_landmarks,
                              get_torch_device,
                              load_safetensors,
                              )
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (convert_ldm_unet_checkpoint,
                                                                    convert_ldm_vae_checkpoint)
from .hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo, FanEncoder
from .hellomeme import HMImagePipeline, HMVideoPipeline
from transformers import CLIPVisionModelWithProjection

DEFAULT_PROMPT = '(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture'

def get_models_files():
    checkpoint_files = folder_paths.get_filename_list("checkpoints")

    vae_files = folder_paths.get_filename_list("vae")
    vae_files = ["[vae] " + x for x in vae_files] + \
                ["[checkpoint] " + x for x in checkpoint_files]

    lora_files = folder_paths.get_filename_list("loras")

    return ['SD1.5'] + checkpoint_files, ['same as checkpoint', 'SD1.5 default vae'] + vae_files, ['None'] + lora_files

def append_pipline_weights(pipeline, checkpoint=None, lora=None, vae=None):
    ### load customized checkpoint or lora here:
    ## checkpoints

    raw_stats = None
    if checkpoint and not checkpoint.startswith('SD1.5'):
        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
        if osp.exists(checkpoint_path):
            print("Loading checkpoint from", checkpoint_path)
            if checkpoint_path.endswith('.safetensors'):
                raw_stats = load_safetensors(checkpoint_path)
            else:
                raw_stats = torch.load(checkpoint_path)

            if raw_stats:
                state_dict = convert_ldm_unet_checkpoint(raw_stats,
                                                         pipeline.unet_pre.config if hasattr(pipeline, 'unet_pre')
                                                         else pipeline.unet.config)
                if hasattr(pipeline, 'unet_pre'):
                    pipeline.unet_pre.load_state_dict(state_dict, strict=False)
                pipeline.unet.load_state_dict(state_dict, strict=False)

    if vae and not vae.startswith('SD1.5 default vae'):
        raw_vae_stats = raw_stats
        real_vae_path = ''
        if vae.startswith("[checkpoint] "):
            real_vae_path = folder_paths.get_full_path_or_raise("checkpoints", vae.replace("[checkpoint] ", ""))
        elif vae.startswith("[vae] "):
            real_vae_path = folder_paths.get_full_path_or_raise("vae", vae.replace("[vae] ", ""))
        if osp.isfile(real_vae_path):
            print("Loading vae from", real_vae_path)
            if real_vae_path.endswith('.safetensors'):
                raw_vae_stats = load_safetensors(real_vae_path)
            else:
                raw_vae_stats = torch.load(real_vae_path)

        if raw_vae_stats:
            vae_state_dict = convert_ldm_vae_checkpoint(raw_vae_stats, pipeline.vae.config)
            if hasattr(pipeline, 'vae_decode'):
                pipeline.vae_decode.load_state_dict(vae_state_dict, strict=True)
            else:
                pipeline.vae.load_state_dict(vae_state_dict, strict=True)

    ### lora
    if lora and not lora.startswith('None'):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora)
        if osp.exists(lora_path):
            print("Loading lora from", lora_path)
            pipeline.load_lora_weights(osp.dirname(lora_path), weight_name=osp.basename(lora_path), adapter_name="lora")

class HMImagePipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files, vae_files, lora_files = get_models_files()

        return {
            "optional": {
                "checkpoint": (checkpoint_files, ),
                "lora": (lora_files, ),
                "vae": (vae_files, ),
                "version": (['v1', 'v2'], ),
            }
        }
    RETURN_TYPES = ("HMIMAGEPIPELINE", )
    RETURN_NAMES = ("hm_image_pipeline", )
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"
    def load_pipeline(self, checkpoint=None, lora=None, vae=None, version='v2'):
        dtype = torch.float16
        pipeline = HMImagePipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version)

        append_pipline_weights(pipeline, checkpoint=checkpoint, lora=lora, vae=vae)

        pipeline.insert_hm_modules(version=version, dtype=dtype)
        
        return (pipeline, )


class HMVideoPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files, vae_files, lora_files = get_models_files()

        return {
            "optional": {
                "checkpoint": (checkpoint_files, ),
                "lora": (lora_files, ),
                "vae": (vae_files, ),
                "version": (['v1', 'v2'], ),
            }
        }

    RETURN_TYPES = ("HMVIDEOPIPELINE",)
    RETURN_NAMES = ("hm_video_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"

    def load_pipeline(self, checkpoint=None, lora=None, vae=None, version='v2'):
        dtype = torch.float16
        pipeline = HMVideoPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version)

        append_pipline_weights(pipeline, checkpoint=checkpoint, lora=lora, vae=vae)

        pipeline.insert_hm_modules(version=version, dtype=dtype)

        return (pipeline,)


class HMFaceToolkitsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
            }
        }

    RETURN_TYPES = ("FACE_TOOLKITS",)
    RETURN_NAMES = ("face_toolkits",)
    FUNCTION = "load_face_toolkits"
    CATEGORY = "hellomeme"
    def load_face_toolkits(self, gpu_id):
        dtype = torch.float16
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                'h94/IP-Adapter', subfolder='models/image_encoder')
        image_encoder.to(dtype=dtype).cpu()
        pd_fpg_motion = FanEncoder.from_pretrained("songkey/pd_fgc_motion")
        pd_fpg_motion.to(dtype=dtype).cpu()
        return (
            dict(
                 device=get_torch_device(gpu_id),
                 dtype=dtype,
                 pd_fpg_motion=pd_fpg_motion,
                 face_aligner=HelloCameraDemo(face_alignment_module=HelloFaceAlignment(gpu_id=gpu_id), reset=False),
                 harkit_bs=HelloARKitBSPred(gpu_id=gpu_id),
                 h3dmm=Hello3DMMPred(gpu_id=gpu_id),
                 image_encoder=image_encoder
            ),
        )

class CropPortrait:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_toolkits": ("FACE_TOOLKITS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_portrait"
    CATEGORY = "hellomeme"

    def crop_portrait(self, image, face_toolkits):
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


class GetFaceLandmarks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FACELANDMARKS222",)
    RETURN_NAMES = ("landmarks",)
    FUNCTION = "get_face_landmarks"
    CATEGORY = "hellomeme"

    def get_face_landmarks(self, face_toolkits, images):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        frame_num = len(frame_list)
        assert frame_num > 0
        _, landmark_list = det_landmarks(face_toolkits['face_aligner'], frame_list)
        assert len(frame_list) == frame_num

        return (torch.from_numpy(landmark_list).float(), )


class GetDrivePose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_POSE",)
    RETURN_NAMES = ("drive_pose",)
    FUNCTION = "get_drive_pose"
    CATEGORY = "hellomeme"
    def get_drive_pose(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()

        rot_list, trans_list = get_drive_pose(face_toolkits, frame_list, landmarks, save_size=512)
        return (dict(rot=np.stack(rot_list), trans=np.stack(trans_list)), )


class GetDriveExpression:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_EXPRESSION",)
    RETURN_NAMES = ("drive_exp",)
    FUNCTION = "get_drive_expression"
    CATEGORY = "hellomeme"
    def get_drive_expression(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()
        exp_dict = get_drive_expression(face_toolkits, frame_list, landmarks)
        return (exp_dict, )


class GetDriveExpression2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_EXPRESSION2",)
    RETURN_NAMES = ("drive_exp2",)
    FUNCTION = "get_drive_expression"
    CATEGORY = "hellomeme"
    def get_drive_expression(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()
        exp_dict = get_drive_expression_pd_fgc(face_toolkits, frame_list, landmarks)
        return (exp_dict, )


class HMPipelineImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hm_image_pipeline": ("HMIMAGEPIPELINE",),
                "face_toolkits": ("FACE_TOOLKITS",),
                "ref_image": ("IMAGE",),
                "drive_pose": ("DRIVE_POSE",),
                "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT}),
                "negative_prompt": ("STRING", {"default": ''}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
            },
                "optional": {
                    "drive_exp": ("DRIVE_EXPRESSION", {"default": None},),
                    "drive_exp2": ("DRIVE_EXPRESSION2", {"default": None},),
                }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
               hm_image_pipeline,
               face_toolkits,
               ref_image,
               drive_pose,
               drive_exp=None,
               drive_exp2=None,
               trans_ratio='0.0',
               prompt=DEFAULT_PROMPT,
               negative_prompt='',
               steps=25,
               seed=0,
               guidance_scale=2.0,
               gpu_id=0
               ):
        device = get_torch_device(gpu_id)

        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        assert len(faces) > 0
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        _, ref_trans = face_toolkits['h3dmm'].forward_params(image_np, ref_landmark)

        drive_rot, drive_trans = drive_pose['rot'], drive_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, 512, trans_ratio)
        drive_params = dict(condition=condition.unsqueeze(0).to(dtype=torch.float16, device='cpu'))

        if isinstance(drive_exp, dict):
            drive_params.update(drive_exp)
        if isinstance(drive_exp2, dict):
            drive_params.update(drive_exp2)

        generator = torch.Generator().manual_seed(seed)

        result_img, latents = hm_image_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
            output_type='np'
        )
        return (torch.from_numpy(np.clip(result_img[0], 0, 1)), dict(samples=latents), )


class HMPipelineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required":{
                        "hm_video_pipeline": ("HMVIDEOPIPELINE",),
                        "face_toolkits": ("FACE_TOOLKITS",),
                        "ref_image": ("IMAGE",),
                        "drive_pose": ("DRIVE_POSE",),
                        "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "patch_overlap": ("INT", {"default": 4, "min": 0, "max": 5}),
                        "prompt": ("STRING", {"default": DEFAULT_PROMPT}),
                        "negative_prompt": ("STRING", {"default": ''}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                          "tooltip": "The number of steps used in the denoising process."}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                         "tooltip": "The random seed used for creating the noise."}),
                        "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                        "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
                     },
                    "optional": {
                        "drive_exp": ("DRIVE_EXPRESSION", {"default": None},),
                        "drive_exp2": ("DRIVE_EXPRESSION2", {"default": None},),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
                hm_video_pipeline,
                face_toolkits,
                ref_image,
                drive_pose,
                drive_exp=None,
                drive_exp2=None,
                trans_ratio=0.0,
                patch_overlap=4,
                prompt=DEFAULT_PROMPT,
                negative_prompt="",
                steps=25,
                seed=0,
                guidance_scale=2.0,
                gpu_id=0
        ):
        device = get_torch_device(gpu_id)

        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        face_toolkits['face_aligner'].reset_track()
        assert len(faces) > 0
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        _, ref_trans = face_toolkits['h3dmm'].forward_params(image_np, ref_landmark)

        generator = torch.Generator().manual_seed(seed)

        drive_rot, drive_trans = drive_pose['rot'], drive_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, 512, trans_ratio)
        drive_params = dict(condition=condition.unsqueeze(0).to(dtype=torch.float16, device='cpu'))

        if isinstance(drive_exp, dict):
            drive_params.update(drive_exp)
        if isinstance(drive_exp2, dict):
            drive_params.update(drive_exp2)

        res_frames, latents = hm_video_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            chunk_overlap=patch_overlap,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
            output_type='np'
        )
        res_frames = [np.clip(x[0], 0, 1) for x in res_frames]
        latents = rearrange(latents[0], 'c f h w -> f c h w')

        return (torch.from_numpy(np.array(res_frames)), dict(samples=latents), )


NODE_CLASS_MAPPINGS = {
    "HMImagePipelineLoader": HMImagePipelineLoader,
    "HMVideoPipelineLoader": HMVideoPipelineLoader,
    "HMFaceToolkitsLoader": HMFaceToolkitsLoader,
    "HMPipelineImage": HMPipelineImage,
    "HMPipelineVideo": HMPipelineVideo,
    "CropPortrait": CropPortrait,
    "GetFaceLandmarks": GetFaceLandmarks,
    "GetDrivePose": GetDrivePose,
    "GetDriveExpression": GetDriveExpression,
    "GetDriveExpression2": GetDriveExpression2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HMImagePipelineLoader": "Load HelloMemeImage Pipeline",
    "HMVideoPipelineLoader": "Load HelloMemeVideo Pipeline",
    "HMFaceToolkitsLoader": "Load Face Toolkits",
    "HMPipelineImage": "HelloMeme Image Pipeline",
    "HMPipelineVideo": "HelloMeme Video Pipeline",
    "CropPortrait": "Crop Portrait",
    "GetFaceLandmarks": "Get Face Landmarks",
    "GetDrivePose": "Get Drive Pose",
    "GetDriveExpression": "Get Drive Expression",
    "GetDriveExpression2": "Get Drive Expression V2",
}
