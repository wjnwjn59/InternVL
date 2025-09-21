import os
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import torch
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
# from internvl.model_lora_embedding.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
import argparse
import json
from tqdm import tqdm
from datetime import datetime
import pytz

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

def load_model_and_tokenizer(args):
    if args.auto:
        config = InternVLChatConfig.from_pretrained(args.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
    kwargs = {'device_map': device_map} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, 
                                              use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    return model, tokenizer

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

model_version = "internvl2_5_2b_groundingnias_dynamic_res_2nd_finetune_lora_vla_pv5_modified_llm_lora_rank_512_epochs_50_backbone_lora_8_bs_2_mdb_9_pdbs_1"
# model_version = "internvl2_5_2b_dynamic_res_2nd_finetune_full_vla_pv5_epochs_50_backbone_lora_8_bs_4_mdb_9_pdbs_1"
# model_version = "internvl2_5_2b_dynamic_res_2nd_finetune_lora_vla_pv5_modified_llm_lora_rank_256_epochs_30_backbone_lora_8_2_bs_4_mdb_9_pdbs_1"
# model_version = "InternVL2_5-8B"
# checkpoint_path = f'work_dirs/internvl_chat_v2_5/{model_version}'
checkpoint_path = f"/media/oem/Research/Thang/internvl_vla_lora/{model_version}"
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=checkpoint_path)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--num-beams', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--out-dir', type=str, default='internvl_vla_dataset')
parser.add_argument('--few-shot', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--load-in-8bit', action='store_true')
parser.add_argument('--load-in-4bit', action='store_true')
parser.add_argument('--auto', action='store_true')
args = parser.parse_args()

model, tokenizer = load_model_and_tokenizer(args)

file_path = f'{args.out_dir}/test_vla_system_prompt_v5.jsonl'
root_image = f'{args.out_dir}/'

# Create result subfolder based on model checkpoint
folder_name = "inference_results_pretrained" if 'pretrained' in args.checkpoint else "inference_results"
result_folder = os.path.join(args.out_dir, folder_name)

# Create the folder if it doesn't exist
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Create the result JSONL file if it doesn't exist
result_file = os.path.join(result_folder, f'{model_version}_{datetime.now(pytz.utc).strftime("%Y%m%d%H%M%S")}.jsonl')

# If the result file doesn't exist, create it and add headers if needed
if not os.path.exists(result_file):
    with open(result_file, 'w') as result_f:
        result_f.write('')

with open(file_path, 'r') as file:
    for i, line in enumerate(tqdm(file)):
        entry = json.loads(line)
        image_path = root_image + entry['image']
        pixel_values = load_image(image_path).to(torch.bfloat16).cuda()
        question = entry['conversations'][0]['value']
        response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=1024, 
                                                                      pad_token_id=tokenizer.eos_token_id,
                                                                      do_sample=False))
        

        
        content = {
            'id': i,
            'image': image_path,
            'send_prompt': question,
            'prediction': response,
            'width': entry['width'],
            'height': entry['height']
        }
        
        with open(result_file, 'a') as result_f:
            result_f.write(json.dumps(content) + "\n")
