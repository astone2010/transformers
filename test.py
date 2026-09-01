"""
Some bits are from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
"""

from huggingface_hub import hf_hub_download
from accelerate.utils import set_module_tensor_to_device, compute_module_sizes
from accelerate import init_empty_weights
from convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, check_quantized_param
from diffusers import FluxTransformer2DModel, FluxPipeline
import safetensors.torch
import gc
import torch

dtype = torch.bfloat16
is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
ckpt_path = hf_hub_download("shauray/flux.1-dev-uncensored-nf4", filename="diffusion_pytorch_model.safetensors")
original_state_dict = safetensors.torch.load_file(ckpt_path)

with init_empty_weights():
    config = FluxTransformer2DModel.load_config("shauray/flux.1-dev-uncensored-nf4")
    model = FluxTransformer2DModel.from_config(config).to(dtype)
    expected_state_dict_keys = list(model.state_dict().keys())

_replace_with_bnb_linear(model, "nf4")

# Determine the device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

for param_name, param in original_state_dict.items():
    if param_name not in expected_state_dict_keys:
        continue
    
    is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
    if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
        param = param.to(dtype)
    
    if not check_quantized_param(model, param_name):
        set_module_tensor_to_device(model, param_name, device=device, value=param)
    else:
        create_quantized_param(
            model, param, param_name, target_device=device, state_dict=original_state_dict, pre_quantized=True
        )

del original_state_dict
gc.collect()

# Ensure model is on GPU
if torch.cuda.is_available():
    model = model.to(device)
    print(f"Model moved to GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024:.2f}GB")

print(f"Model size: {compute_module_sizes(model)[''] / 1024 / 1024:.2f}MB")

pipe = FluxPipeline.from_pretrained("black-forest-labs/flux.1-dev", transformer=model, torch_dtype=dtype)
pipe.to(device)  # Ensure pipeline is on GPU
# pipe.load_lora_weights(
#     "F:\\Comfyui\\models\\loras\\bd_flux_lora_v1.1_000001750.safetensors", 
#     weight_name="bd_flux_lora_v1.1_000001750.safetensors"
# )
prompt = "holding a realistic tiny t rex"
image = pipe(prompt, guidance_scale=3.5, num_inference_steps=20, generator=torch.manual_seed(0)).images[0]
image.save("trex-realistic.png")
