import torch
import toml
import os
import glob
import comfy.sd
import comfy.utils
import comfy.lora
import folder_paths
import hashlib
from server import PromptServer
from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
from comfy.cli_args import args
from nodes import LoraLoader
from .utils import extractLoras, getNewTomlnameExt, load_lora_for_models, runNegpip

routes = PromptServer.instance.routes

@routes.post('/mittimi_path')
async def my_function(request):
    the_data = await request.post()
    LoadSetParamMittimi.handle_my_message(the_data)
    return web.json_response({})

my_directory_path = os.path.dirname((os.path.abspath(__file__)))
presets_directory_path = os.path.join(my_directory_path, "presets")
preset_list = []
tmp_list = []
tmp_list += glob.glob(f"{presets_directory_path}/**/*.toml", recursive=True)
for l in tmp_list:
    preset_list.append(os.path.relpath(l, presets_directory_path))
if len(preset_list) > 1: preset_list.sort()
vae_new_list = folder_paths.get_filename_list("vae")
vae_new_list.insert(0,"Use_merged_vae")

class LoadSetParamMittimi:
    preset_data_cache = {}  # Added cache for preset data

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (preset_list,),
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "ClipNum": ("INT", {"default": -1, "min": -10, "max": -1}),
                "vae": (vae_new_list,),
                "PosPromptA": ("STRING", {"multiline": True}),
                "PosPromptB": ("STRING", {"multiline": True}),
                "PosPromptC": ("STRING", {"multiline": True}),
                "NegPromptA": ("STRING", {"multiline": True}),
                "NegPromptB": ("STRING", {"multiline": True}),
                "NegPromptC": ("STRING", {"multiline": True}),
                "Width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "Height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "BatchSize": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "Steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                # "CFG": ("FLOAT", {"default": 6.0, "step": 0.1, "display": "slider"}),
                "CFG": ("FLOAT", {"default": 6.0, "step": 0.1, "display": "slider"}),
                # "CFG": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
                "SamplerName": (comfy.samplers.KSampler.SAMPLERS,),
                "Scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "Seed": ("INT", {"default": 1}),
            },
            "hidden": {"node_id": "UNIQUE_ID"}
        }

    @classmethod
    def parse_cfg_range(cls, cfg_str: str):
        try:
            range_part, default_str = cfg_str.split(',', 1)
            cfg_min, cfg_max = map(float, range_part.split('-'))
            cfg_default = float(default_str.strip())  # Add .strip()
            return {
                "min": cfg_min,
                "max": cfg_max,
                "default": cfg_default,
                "step": 0.1
            }
        except:
            return {"min": 6.0, "max": 7.0, "default": 6.0}
    @classmethod
    def load_from_preset(cls, preset_path: str):
        """Load CFG range and default from preset.toml."""
        with open(preset_path, "r") as file:
            preset_data = toml.load(file)
        cfg_str = preset_data.get("CFG", "6.0-7.0,6.0")
        return cls.parse_cfg_range(cfg_str)

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LATENT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "PDATA", )
    RETURN_NAMES = ("model", "clip", "vae", "positive_prompt", "negative_prompt", "positive_prompt_text", "negative_prompt_text", "Latent", "Steps", "CFG", "sampler_name", "scheduler", "seed", "parameters_data", )
    FUNCTION = "loadAndSettingParameters03"
    CATEGORY = "mittimiTools"
    NAME = "name"

    def loadAndSettingParameters03(self, checkpoint, ClipNum, vae, PosPromptA, PosPromptB, PosPromptC, NegPromptA, NegPromptB, NegPromptC, Width, Height, BatchSize, Steps, CFG, SamplerName, Scheduler, Seed, node_id, preset=[], ):
        # Apply preset CFG default if needed
        preset_cfg = {}
        if node_id in LoadSetParamMittimi.preset_data_cache:
            preset_cfg = LoadSetParamMittimi.preset_data_cache[node_id].get("CFG", {})
            if CFG == 6.0:  # Override default if needed
                CFG = preset_cfg.get("default", CFG)

        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        out3 = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        re_ckpt = out3[0]
        re_vae = out3[2]
        sha256_hash = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        model_hash = sha256_hash.hexdigest()[:10]
        re_clip = out3[1].clone()
        re_clip.clip_layer(ClipNum)
        re_ckpt, re_clip = runNegpip(self, re_ckpt, re_clip)
        if (vae != "Use_merged_vae"):
            vae_path = folder_paths.get_full_path("vae", vae)
            sd = comfy.utils.load_torch_file(vae_path)
            re_vae = comfy.sd.VAE(sd=sd)
        Latent = torch.zeros([BatchSize, 4, Height // 8, Width // 8], device=self.device)
        poscombtxt = PosPromptA + PosPromptB + PosPromptC
        negcombtxt = NegPromptA + NegPromptB + NegPromptC
        poslora = []
        postxt, poslora = extractLoras(poscombtxt)
        if len(poslora):
            for lora in poslora:
                if lora['vector']:
                    re_ckpt, re_clip, populated_vector = load_lora_for_models(re_ckpt, re_clip, lora['fullpath'], lora['strength'], lora['strength'], Seed, lora['vector'])
                else:
                    re_ckpt, re_clip = LoraLoader().load_lora(re_ckpt, re_clip, lora['path'], lora['strength'], lora['strength'])
        postokens = re_clip.tokenize(postxt)
        negtokens = re_clip.tokenize(negcombtxt)
        posoutput = re_clip.encode_from_tokens(postokens, return_pooled=True, return_dict=True)
        negoutput = re_clip.encode_from_tokens(negtokens, return_pooled=True, return_dict=True)
        poscond = posoutput.pop("cond")
        negcond = negoutput.pop("cond")
        parameters_data = []
        parameters_data.append( {
            'checkpoint':checkpoint,
            'hash':model_hash,
            'clip':ClipNum,
            'vae':vae,
            'posA':PosPromptA,
            'posB':PosPromptB,
            'posC':PosPromptC,
            'negA':NegPromptA,
            'negB':NegPromptB,
            'negC':NegPromptC,
            'width':Width,
            'height':Height,
            'batchsize':BatchSize,
            'step':Steps,
            'cfg': CFG,
            'cfg_min': preset_cfg.get('min', 6.0),
            'cfg_max': preset_cfg.get('max', 7.0),
            'sampler':SamplerName,
            'scheduler':Scheduler,
            'seed':Seed,
            } )
        return(re_ckpt, re_clip, re_vae, [[poscond, posoutput]], [[negcond, negoutput]], postxt, negcombtxt, {"samples":Latent}, Steps, CFG, SamplerName, Scheduler, Seed, parameters_data, )
    @classmethod
    def handle_my_message(cls, d):
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r') as f:
            preset_data = toml.load(f)

        # Parse CFG and prepare dynamic range
        if "CFG" in preset_data:
            try:
                cfg_obj = cls.parse_cfg_range(preset_data["CFG"])
                preset_data["CFG"] = {
                    "min": cfg_obj["min"],
                    "max": cfg_obj["max"],
                    "default": cfg_obj["default"]
                }
            except:
                preset_data["CFG"] = {"min": 6.0, "max": 7.0, "default": 6.0}

        PromptServer.instance.send_sync("my.custom.message", {
            "message": preset_data,
            "node": d['node_id']
        })
class CombineParamDataMittimi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "param1": ("PDATA", ),
            },
            "optional": {
                "param2": ("PDATA", ),
            },
        }
    RETURN_TYPES = ("PDATA", )
    RETURN_NAMES = ("parameters_data", )
    FUNCTION = "combineparamdataMittimi"
    CATEGORY = "mittimiTools"
    def combineparamdataMittimi(self, param1, param2=None, ):
        return_param = param1[:]
        if param2:
            return_param += param2[:]
        return(return_param, )

class SaveImageParamMittimi:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        file_type = ['PNG', 'WEBP']
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "type": (file_type, ),
                "level_or_method": ("INT", {"default":6, "min":1, "max":9}),
                "lossless": ("BOOLEAN", {"default":True}),
            },
            "optional": {
                "quality": ("INT", {"default": 80, "min": 1, "max": 100}),
                "parameters_data": ("PDATA", ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveimageMittimi"
    CATEGORY = "mittimiTools"

    def saveimageMittimi(self, images, filename_prefix, type, level_or_method, lossless, quality=80, parameters_data=None, prompt=None, extra_pnginfo=None, ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        bseed = 0
        if parameters_data:
            bseed = parameters_data[0]['seed'] if "seed" in parameters_data[0] else 0
        bseed = "_"+str(bseed) if bseed>0 else ""
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            parameters_text = ""
            if parameters_data is not None:
                param_counter = 0
                for pd in parameters_data:
                    if param_counter > 0: parameters_text += "\n"
                    parameters_text += f"{pd['posA']+pd['posB']+pd['posC']}\nNegative prompt: {pd['negA']+pd['negB']+pd['negC']}\nSteps: {pd['step']}, Sampler: {pd['sampler']}, Scheduler: {pd['scheduler']}, CFG Scale: {pd['cfg']}, Seed: {pd['seed']}, Batch count: {batch_number}, Size: {pd['width']}x{pd['height']}, Model hash: {pd['hash']}, Model: {pd['checkpoint']}, Clip: {pd['clip']}, VAE: {pd['vae']}"
                    param_counter += 1
                parameters_text += ", Version: ComfyUI"
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            rseed = bseed if batch_number == 0 else f"{bseed}_{batch_number:03}"
            if type =="WEBP":
                metadata = img.getexif()
                metadata[270] = f"parameters:{parameters_text}"
                if prompt is not None:
                    metadata[272] = "prompt:{}".format(json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata[271] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                file = f"{filename_with_batch_num}_{counter:05}{rseed}.webp"
                method = level_or_method if level_or_method <7 else 6
                img.save(os.path.join(full_output_folder, file), "webp", exif=metadata, lossless=lossless, quality=quality, method=method)
            else:
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                metadata.add_text("parameters", parameters_text)
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                file = f"{filename_with_batch_num}_{counter:05}{rseed}.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=level_or_method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return { "ui": { "images": results } }

class SaveParamToPresetMittimi:
    @classmethod
    def INPUT_TYPES(s):
        savetype_list = ["new save", "overwrite save"]
        return {
            "required": {
                "param": ("PDATA", ),
                "tomlname": ("STRING", {"default": "new_preset"}),
                "savetype": (savetype_list,),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveparamtopresetMittimi"
    CATEGORY = "mittimiTools"

    def saveparamtopresetMittimi(self, param, tomlname, savetype, ):
        # Get current CFG value from parameters
        cfg = param[0]['cfg']
        cfg_min = param[0].get('cfg_min', 6.0)
        cfg_max = param[0].get('cfg_max', 7.0)
        cfg_str = f"{cfg_min}-{cfg_max},{cfg}"
        tomltext = f"CheckpointName = \"{param[0]['checkpoint']}\"\n"
        tomltext += f"ClipSet = {param[0]['clip']}\n"
        tomltext += f"VAE = \"{param[0]['vae']}\"\n"
        tomltext += f"PositivePromptA = \"{param[0]['posA']}\"\n"
        tomltext += f"PositivePromptB = \"{param[0]['posB']}\"\n"
        tomltext += f"PositivePromptC = \"{param[0]['posC']}\"\n"
        tomltext += f"NegativePromptA = \"{param[0]['negA']}\"\n"
        tomltext += f"NegativePromptB = \"{param[0]['negB']}\"\n"
        tomltext += f"NegativePromptC = \"{param[0]['negC']}\"\n"
        tomltext += f"Width = {param[0]['width']}\n"
        tomltext += f"Height = {param[0]['height']}\n"
        tomltext += f"BatchSize = {param[0]['batchsize']}\n"
        tomltext += f"Steps = {param[0]['step']}\n"
        tomltext += f"CFG = \"{cfg_str}\"\n"
        tomltext += f"SamplerName = \"{param[0]['sampler']}\"\n"
        tomltext += f"Scheduler = \"{param[0]['scheduler']}\"\n"

        tomlnameExt = getNewTomlnameExt(tomlname, presets_directory_path, savetype)
        check_folder_path = os.path.dirname(f"{presets_directory_path}/{tomlnameExt}")
        os.makedirs(check_folder_path, exist_ok=True)
        with open(f"{presets_directory_path}/{tomlnameExt}", mode='w') as f:
            f.write(tomltext)
        return()

NODE_CLASS_MAPPINGS = {
    "LoadSetParamMittimi": LoadSetParamMittimi,
    "SaveImageParamMittimi": SaveImageParamMittimi,
    "CombineParamDataMittimi": CombineParamDataMittimi,
    "SaveParamToPresetMittimi": SaveParamToPresetMittimi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSetParamMittimi": "LoadSetParameters",
    "SaveImageParamMittimi": "SaveImageWithParamText",
    "CombineParamDataMittimi": "CombineParamData",
    "SaveParamToPresetMittimi": "SaveParamToPreset",
}
