# TensorRT Extension

Use this extension to generate optimized engines and enable the best performance on NVIDIA RTX GPUs with TensorRT. Please follow the instructions below to set everything up.

## Set Up

1. Click on the "Generate Default Engines" button. This step can take 2-10 min depending on your GPU. You can generate engines for other combinations. 
2. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
3. Back in the main UI, select the TRT model from the sd_unet dropdown menu at the top of the page.
4. You can now start generating images accelerated by TRT. If you need to create more Engines, go to the TensorRT tab.

Happy prompting!

## More Information

TensorRT uses optimized engines for specific resolutions and batch sizes. You can generate as many optimized engines as desired. Types:

- The "Export Default Engines" selection adds support for resolutions between 512x512 and 768x768 for Stable Diffusion 1.5 and 768x768 to 1024x1024 for SDXL with batch sizes 1 to 4.
- Static engines support a single specific output resolution and batch size.
- Dynamic engines support a range of resolutions and batch sizes, at a small cost in performance. Wider ranges will use more VRAM. 

---

Each preset can be adjusted with the "Advanced Settings" option.

For more information, please visit the TensorRT Extension GitHub page [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt).


## Appendix 1: Pluggable Lora

This version supplies pluggable lora. It has features as follows:
1. Use cupy-cuda11x package to speed up the large matrix calculation.
2. Separate lora into three parts: original unet, common base unet and lora. We try to make it faster, so the common base unet contains those nodes that may be changed during applying loras.
3. If you dislike, just replace upon part with those scripts in branch lora_v2

## Appendix 2: ControlNet 

the ControlNet is based on the [controlnet](https://github.com/Mikubill/sd-webui-controlnet). When you want to use tensort controlnet, do as follows:
1. Download [controlnet](https://github.com/Mikubill/sd-webui-controlnet), and put it into folder extension;
2. Move the "controlnet/controlnet.py" and "controlnet/hook.py" into "extensions/sd-webui-controlnet/scripts" and overwrite the old scripts. 
3. Move the yaml "controlnet/cldm_v15.yaml" into folder "extensions/sd-webui-controlnet/models" and overwrite existing yaml file;
4. Move the "examples/engines" into root folder where launch.py exists. the "engines" script is used to load engines and is imported by contronet.py.

## Appendix 3: refresh
After generate new unet engine, please reload.