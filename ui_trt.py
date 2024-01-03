import os
import json
from modules import sd_models, shared
import gradio as gr

from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow

from exporter import export_onnx, export_trt, onnx_to_refit_delta
from utilities import PIPELINE_TYPE, Engine
from models import make_OAIUNetXL, make_OAIUNet, make_ControlNet
import logging
import gc
import torch
from model_manager import modelmanager, cc_major, TRT_MODEL_DIR
from time import sleep
from collections import defaultdict
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)


def get_version_from_model(sd_model):
    if sd_model.is_sd1:
        return "1.5"
    if sd_model.is_sd2:
        return "2.1"
    if sd_model.is_sdxl:
        return "xl-1.0"


def export_unet_to_trt(
    batch_min,
    batch_opt,
    batch_max,
    height_min,
    height_opt,
    height_max,
    width_min,
    width_opt,
    width_max,
    token_count_min,
    token_count_opt,
    token_count_max,
    force_export,
    static_shapes,
    preset,
    controlnet,
    trt_name: str
):
    print(f"==== export_unet_to_trt params: \n "
          f"batch_min: {batch_min}, batch_opt: {batch_opt}, batch_max: {batch_max}\n"
          f"height_min: {height_min}, height_opt: {height_opt}, height_max: {height_max}\n"
          f"width_min: {width_min}, width_opt: {width_opt}, width_max: {width_max}\n"
          f"token_count_min: {token_count_min}, token_count_opt: {token_count_opt}, token_count_max: {token_count_max}\n"
          f"force_export: {force_export}, static_shapes: {static_shapes}, preset: {preset}, controlnet: {controlnet},\n"
          f"trt_name: {trt_name}")

    if preset == "Default":
        (
            batch_min,
            batch_opt,
            batch_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            token_count_min,
            token_count_opt,
            token_count_max,
        ) = export_default_unet_to_trt()
    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        print("FP16 has been disabled because your GPU does not support it.")

    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name # todo: 这里要不要加上controlnet的标志呢？
    if controlnet:
        model_name += "-control"
    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name, model_hash)

    print(f"Exporting {model_name} to TensorRT")

    timing_cache = modelmanager.get_timing_cache()

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT

    min_textlen = (token_count_min // 75) * 77
    opt_textlen = (token_count_opt // 75) * 77
    max_textlen = (token_count_max // 75) * 77
    if static_shapes:
        min_textlen = max_textlen = opt_textlen
        batch_min = batch_max = batch_opt
        width_min = width_max = width_opt
        height_min = height_max = height_opt

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(
            version, pipeline, "cuda", False, batch_max, opt_textlen, max_textlen
        )
        diable_optimizations = True
    else:
        modelobj = make_OAIUNet(
            version,
            pipeline,
            "cuda",
            False,
            batch_max,
            opt_textlen,
            max_textlen,
            controlnet,
            static_shapes
        )
        diable_optimizations = False

    profile = modelobj.get_input_profile(
        batch_min,
        batch_opt,
        batch_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        static_shapes,
    )
    print(profile)

    if not os.path.exists(onnx_path) or force_export:
        print("No ONNX file found. Exporting ONNX...")
        gr.Info("No ONNX file found. Exporting ONNX...  Please check the progress in the terminal.")
        export_onnx(
            onnx_path,
            modelobj,
            profile=profile,
            diable_optimizations=diable_optimizations,
        )
        print("Exported to ONNX.")

    if "" == trt_name:
        trt_engine_filename, trt_path = modelmanager.get_trt_path(
            model_name, model_hash, profile, static_shapes
        )
    else:
        if not trt_name.endswith(".trt"):
            trt_name += ".trt"
        trt_engine_filename, trt_path = trt_name, os.path.join(TRT_MODEL_DIR, trt_name)
    # todo: 截断
    if not os.path.exists(trt_path) or force_export:
        print("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gr.Info("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gc.collect()
        torch.cuda.empty_cache()
        ret = export_trt(
            trt_path,
            onnx_path,
            timing_cache,
            profile=profile,
            use_fp16=not use_fp32,
        )
        if ret:
            return "## Export Failed due to unknown reason. See shell for more information. \n"

        print("TensorRT engines has been saved to disk.")
        modelmanager.add_entry(
            model_name,
            trt_engine_filename,
            profile,
            static_shapes,
            fp32=use_fp32,
            inpaint=is_inpaint,
            refit=True,
            vram=0,
            unet_hidden_dim=unet_hidden_dim,
            lora=False,
        )
    else:
        print("TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.")

    return "## Exported Successfully \n"


def export_lora_to_trt(base_model_name, profile_info, lora_name, force_export, no_refit):
    print(f"==== base_model_name: {base_model_name}, profile_info: {profile_info}, lora_name: {lora_name}")

    # 默认的unet profile
    all_models = modelmanager.available_models()
    base_model_config = all_models[base_model_name]
    unet_profile = None
    profile_idx = int(profile_info.split("_")[0])
    static_shapes = False
    if profile_idx < len(base_model_config):
        static_shapes = base_model_config[profile_idx]['config'].static_shapes
        unet_profile = base_model_config[profile_idx]['config'].profile

    if unet_profile is None:
        raise ValueError("Selected ControlUnet Config doesn't exist in model.json.")

    use_controlnet = len(unet_profile.keys()) > 3
    batch_max = unet_profile["sample"][2][0]
    opt_len, max_len = unet_profile["encoder_hidden_states"][-2][1], unet_profile["encoder_hidden_states"][-1][1]

    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        print("FP16 has been disabled because your GPU does not support it.")
    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    if use_controlnet:
        model_name += "-control"
    base_name = f"{model_name}"  # _{model_hash}

    available_lora_models = get_lora_checkpoints()
    lora_name = lora_name.split(" ")[0]
    lora_model = available_lora_models[lora_name]

    onnx_base_filename, onnx_base_path = modelmanager.get_onnx_path(
        model_name, model_hash
    )
    onnx_lora_filename, onnx_lora_path = modelmanager.get_onnx_path(
        lora_name, base_name
    )

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(version, pipeline, "cuda", False, 1, 77, 77)
        diable_optimizations = True
    else:
        print(f"===== len(unet_profile.keys()): {len(unet_profile.keys())}, use_controlnet: {use_controlnet}")
        modelobj = make_OAIUNet(
            version=version,
            pipeline=pipeline,
            device="cuda",
            verbose=False,
            max_batch_size=batch_max,
            text_optlen=opt_len,
            text_maxlen=max_len,
            controlnet=use_controlnet,
            static_shape=static_shapes
        )
        diable_optimizations = False

    if not os.path.exists(onnx_lora_path) or force_export:
        print("No ONNX file found. Exporting ONNX...")
        gr.Info("No ONNX file found. Exporting ONNX...  Please check the progress in the terminal.")
        export_onnx(
            onnx_lora_path,
            modelobj,
            profile=unet_profile,
            diable_optimizations=diable_optimizations,
            lora_path=lora_model["filename"],
        )
        print("Exported to ONNX.")

    trt_lora_name = onnx_lora_filename.replace(".onnx", ".trt")
    trt_lora_path = os.path.join(TRT_MODEL_DIR, trt_lora_name)

    available_trt_unet = modelmanager.available_models()
    if len(available_trt_unet[base_name]) == 0:
        return f"## Please export the base model ({base_name}) first."

    if not os.path.exists(onnx_base_path):
        return f"## Please export the base model ({base_name}) first."

    if not os.path.exists(trt_lora_path) or force_export:
        print("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gr.Info("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gc.collect()
        torch.cuda.empty_cache()

        if no_refit:
            timing_cache = modelmanager.get_timing_cache()
            ret = export_trt(
                trt_lora_path,
                onnx_lora_path,
                timing_cache,
                profile=unet_profile,
                use_fp16=not use_fp32,
            )
            if ret:
                return "## Export Failed due to unknown reason. See shell for more information. \n"
        else:
            onnx_to_refit_delta(onnx_base_path, onnx_lora_path, trt_lora_path)

        modelmanager.add_lora_entry(
            base_name,
            lora_name,
            trt_lora_name,
            use_fp32,
            is_inpaint,
            0,
            unet_hidden_dim,
        )

        print("TensorRT engines has been saved to disk.")

    return "## Exported Successfully \n"


def export_default_unet_to_trt():
    is_xl = shared.sd_model.is_sdxl

    batch_min = 1
    batch_opt = 1
    batch_max = 4
    height_min = 768 if is_xl else 512
    height_opt = 1024 if is_xl else 512
    height_max = 1024 if is_xl else 768
    width_min = 768 if is_xl else 512
    width_opt = 1024 if is_xl else 512
    width_max = 1024 if is_xl else 768
    token_count_min = 75
    token_count_opt = 75
    token_count_max = 150

    return (
        batch_min,
        batch_opt,
        batch_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        token_count_min,
        token_count_opt,
        token_count_max,
    )


def load_control_model(control_name):
    control_model_path = os.path.join(shared.cmd_opts.control_dir, control_name + ".pth")

    # 加载模型
    config_path = os.path.join(shared.cmd_opts.control_dir, "cldm_v15.yaml")
    config = OmegaConf.load(config_path)

    def get_state_dict(d):
        return d.get('state_dict', d)

    def get_state_dicts(ckpt_path, location='cuda'):
        _, extension = os.path.splitext(ckpt_path)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(ckpt_path, device=location)
        else:
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
        state_dict = get_state_dict(state_dict)
        print(f'Loaded state_dict from [{ckpt_path}]')
        return state_dict

    state_dicts = get_state_dicts(control_model_path)
    controlnet_config = config["model"]['params']['control_stage_config']
    controlnet_model = instantiate_from_config(controlnet_config)

    controlnet_dicts = {k: state_dicts["control_model." + k] for k in controlnet_model.state_dict()}
    controlnet_model.load_state_dict(controlnet_dicts)
    controlnet_model = controlnet_model.to(torch.device("cuda"))
    controlnet_model.eval()

    return controlnet_model


def export_controlnet_to_trt(base_model_name, profile_info, control_name, force_export):
    print(f"==== base_model_name: {base_model_name}, profile_info: {profile_info}, lora_name: {control_name}")

    # 默认的unet profile
    all_models = modelmanager.available_models()
    base_model_config = all_models[base_model_name]
    unet_profile = None
    profile_idx = int(profile_info.split("_")[0])
    static_shapes = False
    if profile_idx < len(base_model_config):
        static_shapes = base_model_config[profile_idx]['config'].static_shapes
        unet_profile = base_model_config[profile_idx]['config'].profile

    if unet_profile is None:
        raise ValueError("Selected ControlUnet Config doesn't exist in model.json.")
    opt_len, max_len = unet_profile["encoder_hidden_states"][-2][1], unet_profile["encoder_hidden_states"][-1][1]
    print(f"==== opt_len: {opt_len}, max_len: {max_len}")
    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        print("FP16 has been disabled because your GPU does not support it.")
    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    onnx_control_filename, onnx_control_path = modelmanager.get_onnx_path(
        control_name, base_model_name
    )
    timing_cache = modelmanager.get_timing_cache()

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT

    modelobj = make_ControlNet(version, pipeline, "cuda", False, max_len, opt_len, static_shapes)
    diable_optimizations = False
    control_profile = modelobj.get_input_profile_v2(unet_profile)
    controlnet_model = load_control_model(control_name)
    if not os.path.exists(onnx_control_path) or force_export:
        print("No ONNX file found. Exporting ONNX...")
        gr.Info("No ONNX file found. Exporting ONNX...  Please check the progress in the terminal.")
        export_onnx(
            onnx_control_path,
            modelobj,
            profile=control_profile,
            diable_optimizations=diable_optimizations,
            model=controlnet_model
        )
        print("Exported to ONNX.")

    trt_control_name = onnx_control_filename.replace(".onnx", ".trt")
    trt_control_path = os.path.join(TRT_MODEL_DIR, trt_control_name)

    # todo: 截断
    if not os.path.exists(trt_control_path) or force_export:
        print("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gr.Info("Building TensorRT engine... This can take a while, please check the progress in the terminal.")
        gc.collect()
        torch.cuda.empty_cache()
        ret = export_trt(
            trt_control_path,
            onnx_control_path,
            timing_cache,
            profile=control_profile,
            use_fp16=not use_fp32,
        )
        if ret:
            return "## Export Failed due to unknown reason. See shell for more information. \n"

        print("TensorRT engines has been saved to disk.")
        modelmanager.add_controlnet_entry(
            base_model=base_model_name,
            control_name=control_name,
            trt_control_path=trt_control_path,
            fp32=use_fp32,
            inpaint=is_inpaint,
            vram=0,
            unet_hidden_dim=unet_hidden_dim,
        )
    else:
        print("TensorRT engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.")

    return "## Exported Successfully \n"



profile_presets = {
    "512x512 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        512,
        512,
        512,
        512,
        512,
        512,
        75,
        75,
        75,
    ),
    "768x768 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        768,
        768,
        768,
        768,
        768,
        768,
        75,
        75,
        75,
    ),
    "1024x1024 | Batch Size 1 (Static)": (
        1,
        1,
        1,
        1024,
        1024,
        1024,
        1024,
        1024,
        1024,
        75,
        75,
        75,
    ),
    "256x256 - 512x512 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        256,
        512,
        512,
        256,
        512,
        512,
        75,
        75,
        150,
    ),
    "512x512 - 768x768 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        512,
        512,
        768,
        512,
        512,
        768,
        75,
        75,
        150,
    ),
    "768x768 - 1024x1024 | Batch Size 1-4 (Dynamic)": (
        1,
        1,
        4,
        768,
        1024,
        1024,
        768,
        1024,
        1024,
        75,
        75,
        150,
    ),
}


def get_settings_from_version(version):
    static = False
    if version == "Default":
        return *list(profile_presets.values())[-2], static
    if "Static" in version:
        static = True
    return *profile_presets[version], static


def diable_export(version):
    if version == "Default":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

def disable_lora_export(lora):
    if lora is None:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def diable_visibility(hide):
    num_outputs = 8
    out = [gr.update(visible=not hide) for _ in range(num_outputs)]
    return out


def engine_profile_card():
    def get_md_table(
        h_min,
        h_opt,
        h_max,
        w_min,
        w_opt,
        w_max,
        b_min,
        b_opt,
        b_max,
        t_min,
        t_opt,
        t_max,
    ):
        md_table = (
            "|             	|   Min   	|   Opt   	|   Max   	| \n"
            "|-------------	|:-------:	|:-------:	|:-------:	| \n"
            "| Height      	| {h_min} 	| {h_opt} 	| {h_max} 	| \n"
            "| Width       	| {w_min} 	| {w_opt} 	| {w_max} 	| \n"
            "| Batch Size  	| {b_min} 	| {b_opt} 	| {b_max} 	| \n"
            "| Text-length 	| {t_min} 	| {t_opt} 	| {t_max} 	| \n"
        )
        return md_table.format(
            h_min=h_min,
            h_opt=h_opt,
            h_max=h_max,
            w_min=w_min,
            w_opt=w_opt,
            w_max=w_max,
            b_min=b_min,
            b_opt=b_opt,
            b_max=b_max,
            t_min=t_min,
            t_opt=t_opt,
            t_max=t_max,
        )

    available_models = modelmanager.available_models()

    model_md = defaultdict(list)
    loras_md = {}
    for base_model, models in available_models.items():
        for i, m in enumerate(models):
            if m["config"].lora:
                loras_md[base_model] = m.get("base_model", None)
                continue
            if "base_model" in m:
                continue
            s_min, s_opt, s_max = m["config"].profile.get(
                "sample", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            )
            t_min, t_opt, t_max = m["config"].profile.get(
                "encoder_hidden_states", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            )
            profile_table = get_md_table(
                s_min[2] * 8,
                s_opt[2] * 8,
                s_max[2] * 8,
                s_min[3] * 8,
                s_opt[3] * 8,
                s_max[3] * 8,
                max(s_min[0] // 2, 1),
                max(s_opt[0] // 2, 1),
                max(s_max[0] // 2, 1),
                (t_min[1] // 77) * 75,
                (t_opt[1] // 77) * 75,
                (t_max[1] // 77) * 75,
            )

            model_md[base_model].append(profile_table)

    for lora, base_model in loras_md.items():
        model_md[f"{lora} (*{base_model}*)"] = model_md[base_model]

    return model_md


def get_version_from_filename(name):
    if "v1-" in name:
        return "1.5"
    elif "v2-" in name:
        return "2.1"
    elif "xl" in name:
        return "xl-1.0"
    else:
        return "Unknown"


def get_lora_checkpoints(): #TODO
    available_lora_models = {}
    candidates = list(
        shared.walk_files(
            shared.cmd_opts.lora_dir,
            allowed_extensions=[".pt", ".ckpt", ".safetensors"],
        )
    )
    for filename in candidates:
        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            metadata = sd_models.read_metadata_from_safetensors(filename)
            version = get_version_from_filename(metadata.get("ss_sd_model_name"))
        except (AssertionError, TypeError):
            version = "Unknown"
        available_lora_models[name] = {
            "filename": filename,
            "version": version,
        }
    return available_lora_models


def refresh_lora_all():
    unets = get_valid_unet_trt()
    loras = get_valid_lora_checkpoints()
    return unets, loras


def get_valid_lora_checkpoints():
    available_lora_models = get_lora_checkpoints()
    return [
        f"{k} ({v['version']})"
        for k, v in available_lora_models.items()
        if v["version"] == get_version_from_model(shared.sd_model)
        or v["version"] == "Unknown"
    ]


def get_valid_controlnet_checkpoints():
    available_control_models = set()
    candidates = list(shared.walk_files(shared.cmd_opts.control_dir,
                                        allowed_extensions=[".pth"]))
    for filename in candidates:
        name = os.path.splitext(os.path.basename(filename))[0]
        available_control_models.add(name)

    return list(available_control_models)


def get_valid_unet_trt():
    """
    读取model.json文件，并且和目录下对应的进行match，存在的才加入到里面来
    """
    with open(os.path.join(TRT_MODEL_DIR, "model.json"), "r") as f:
        out = json.load(f)

    base_model_names = set()
    for cc, models in out.items():
        for name, configs in models.items():
            for config in configs:
                if os.path.exists(os.path.join(TRT_MODEL_DIR, config["filepath"])) and ("base_model" not in config):
                    base_model_names.add(name)

    return list(base_model_names)


def get_valid_controlunet_trt():
    with open(os.path.join(TRT_MODEL_DIR, "model.json"), "r") as f:
        out = json.load(f)

    base_model_names = set()
    for cc, models in out.items():
        for name, configs in models.items():
            for config in configs:
                if os.path.exists(os.path.join(TRT_MODEL_DIR, config["filepath"])) and ("base_model" not in config):
                    if len(config["config"]["profile"].keys()) > 13:
                        base_model_names.add(name)

    return list(base_model_names)


def select_unet_profile(unet_name):
    with open(os.path.join(TRT_MODEL_DIR, "model.json"), "r") as f:
        out = json.load(f)

    profiles = set()
    for cc, models in out.items():
        for name, configs in models.items():
            if name == unet_name:
                for idx, config in enumerate(configs):
                    profiles.add(str(idx) + "_" + config["filepath"])
                break

    res = gr.Dropdown.update(choices=list(profiles), label="profiles")
    return res


def select_unet_control_profile(unet_name):
    with open(os.path.join(TRT_MODEL_DIR, "model.json"), "r") as f:
        out = json.load(f)

    profiles = set()
    for cc, models in out.items():
        for name, configs in models.items():
            if name == unet_name:
                for idx, config in enumerate(configs):
                    if len(config["config"]["profile"]) == 16:
                        profiles.add(str(idx) + "_" + config["filepath"])
                break
    res = gr.Dropdown.update(choices=list(profiles), label="profiles")
    return res


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="TensorRT Exporter"):
                        gr.Markdown(
                            value="# TensorRT Exporter",
                        )

                        default_version = list(profile_presets.keys())[-2]
                        default_vals = list(profile_presets.values())[-2]
                        version = gr.Dropdown(
                            label="Preset",
                            choices=list(profile_presets.keys()) + ["Default"],
                            elem_id="sd_version",
                            default="Default",
                            value="Default",
                        )

                        with gr.Accordion("Advanced Settings", open=False, visible=False) as advanced_settings:
                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                static_shapes = gr.Checkbox(
                                    label="Use static shapes.",
                                    value=False,
                                    elem_id="trt_static_shapes",
                                )

                            with gr.Column(elem_id="trt_max_batch"):
                                trt_min_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Min batch-size",
                                    value=default_vals[0],
                                    elem_id="trt_min_batch",
                                )

                                trt_opt_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Optimal batch-size",
                                    value=default_vals[1],
                                    elem_id="trt_opt_batch",
                                )
                                trt_max_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Max batch-size",
                                    value=default_vals[2],
                                    elem_id="trt_max_batch",
                                )

                            with gr.Column(elem_id="trt_height"):
                                trt_height_min = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min height",
                                    value=default_vals[3],
                                    elem_id="trt_min_height",
                                )
                                trt_height_opt = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal height",
                                    value=default_vals[4],
                                    elem_id="trt_opt_height",
                                )
                                trt_height_max = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max height",
                                    value=default_vals[5],
                                    elem_id="trt_max_height",
                                )

                            with gr.Column(elem_id="trt_width"):
                                trt_width_min = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Min width",
                                    value=default_vals[6],
                                    elem_id="trt_min_width",
                                )
                                trt_width_opt = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Optimal width",
                                    value=default_vals[7],
                                    elem_id="trt_opt_width",
                                )
                                trt_width_max = gr.Slider(
                                    minimum=256,
                                    maximum=4096,
                                    step=64,
                                    label="Max width",
                                    value=default_vals[8],
                                    elem_id="trt_max_width",
                                )

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count_min = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Min prompt token count",
                                    value=default_vals[9],
                                    elem_id="trt_opt_token_count_min",
                                )
                                trt_token_count_opt = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Optimal prompt token count",
                                    value=default_vals[10],
                                    elem_id="trt_opt_token_count_opt",
                                )
                                trt_token_count_max = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Max prompt token count",
                                    value=default_vals[11],
                                    elem_id="trt_opt_token_count_max",
                                )

                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                force_rebuild = gr.Checkbox(
                                    label="Force Rebuild.",
                                    value=False,
                                    elem_id="trt_force_rebuild",
                                )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            use_controlnet = gr.Checkbox(
                                label="Use ControlNet.",
                                value=False,
                                elem_id="trt_use_controlnet",
                                visible=True
                            )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_name = gr.Textbox(
                                lines=1,
                                placeholder="trt engine name",
                                elem_id="trt_name",
                                visible=True
                            )

                        button_export_unet = gr.Button(
                            value="Export Engine",
                            variant="primary",
                            elem_id="trt_export_unet",
                            visible=False,
                        )

                        button_export_default_unet = gr.Button(
                            value="Export Default Engine",
                            variant="primary",
                            elem_id="trt_export_default_unet",
                            visible=True,
                        )

                        version.change(
                            get_settings_from_version,
                            version,
                            [
                                trt_min_batch,
                                trt_opt_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_opt,
                                trt_height_max,
                                trt_width_min,
                                trt_width_opt,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_opt,
                                trt_token_count_max,
                                static_shapes,
                            ],
                        )
                        version.change(
                            diable_export,
                            version,
                            [button_export_unet, button_export_default_unet, advanced_settings],
                        )

                        static_shapes.change(
                            diable_visibility,
                            static_shapes,
                            [
                                trt_min_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_max,
                                trt_width_min,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_max,
                            ],
                        )

                    with gr.Tab(label="TensorRT LoRA"):
                        gr.Markdown("# Apply LoRA checkpoint to TensorRT model")
                        lora_refresh_button = gr.Button(
                            value="Refresh",
                            variant="primary",
                            elem_id="trt_lora_refresh",
                        )

                        trt_unet_dropdown = gr.Dropdown(
                            choices=get_valid_unet_trt(),
                            elem_id="unet_trt",
                            label="Unet Trt",
                            visible=True
                        )

                        trt_unet_profile_dropdown = gr.Dropdown(
                            choices=[],
                            elem_id="unet_profile",
                            label="Unet Profile",
                            visible=True
                        )

                        trt_lora_dropdown = gr.Dropdown(
                            choices=get_valid_lora_checkpoints(),
                            elem_id="lora_model",
                            label="LoRA Model",
                            default=None,
                        )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_lora_force_rebuild = gr.Checkbox(
                                label="Force Rebuild.",
                                value=False,
                                elem_id="trt_lora_force_rebuild",
                            )
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            no_refit = gr.Checkbox(
                                label="No refit.",
                                value=False,
                                elem_id="trt_lora_no_refit",
                            )

                        button_export_lora_unet = gr.Button(
                            value="Convert to TensorRT",
                            variant="primary",
                            elem_id="trt_lora_export_unet",
                            visible=True,
                        )

                        lora_refresh_button.click(
                            refresh_lora_all,
                            None,
                            [trt_unet_dropdown, trt_lora_dropdown],
                        )
                        trt_lora_dropdown.change(
                            disable_lora_export, trt_lora_dropdown, button_export_lora_unet
                        )


                    with gr.Tab(label="TensorRT ControlNet"):
                        gr.Markdown("# Transfer ControlNet Model to TensorRT model")

                        trt_controlunet_dropdown = gr.Dropdown(
                            choices=get_valid_controlunet_trt(),
                            elem_id="controlunet_trt",
                            label="ControlUnet Trt",
                            default=None,
                        )

                        trt_controlunet_profile_dropdown = gr.Dropdown(
                            choices=[],
                            elem_id="controlunet_profile",
                            label="ControlUnet Profiles",
                        )

                        trt_control_dropdown = gr.Dropdown(
                            choices=get_valid_controlnet_checkpoints(),
                            elem_id="controlnet_model",
                            label="ControlNet Model",
                            default=None,
                        )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_control_force_rebuild = gr.Checkbox(
                                label="Force Rebuild.",
                                value=False,
                                elem_id="trt_control_force_rebuild",
                            )

                        button_export_controlnet = gr.Button(
                            value="Convert to TensorRT",
                            variant="primary",
                            elem_id="trt_export_controlnet",
                            visible=True,
                        )

            with gr.Column(variant="panel"):
                with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"),
                    "r",
                    encoding='utf-8',
                ) as f:
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=True):
                trt_result = gr.Markdown(elem_id="trt_result", value="")

        def get_trt_profiles_markdown():
            profiles_md_string = ""
            for model, profiles in engine_profile_card().items():
                profiles_md_string += f"<details><summary>{model} ({len(profiles)} Profiles)</summary>\n\n"
                for i, profile in enumerate(profiles):
                    profiles_md_string += f"#### Profile {i} \n{profile}\n\n"
                profiles_md_string += "</details>\n"
            profiles_md_string += "</details>\n"
            return profiles_md_string


        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True, variant="compact"):
                button_refresh_profiles = ToolButton(value=refresh_symbol, elem_id="trt_refresh_profiles", visible=True)
                profile_header_md = gr.Markdown(
                    value=f"## Available TensorRT Engine Profiles"
                )
            with gr.Row(equal_height=True):
                trt_profiles_markdown = gr.Markdown(elem_id=f"trt_profiles_markdown", value=get_trt_profiles_markdown())

        trt_unet_dropdown.change(
            select_unet_profile,
            trt_unet_dropdown,
            [trt_unet_profile_dropdown]
        )

        trt_controlunet_dropdown.change(
            select_unet_control_profile,
            trt_controlunet_dropdown,
            [trt_controlunet_profile_dropdown]
        )

        button_refresh_profiles.click(lambda: gr.Markdown.update(value=get_trt_profiles_markdown()), outputs=[trt_profiles_markdown])

        button_export_unet.click(
            export_unet_to_trt,
            inputs=[
                trt_min_batch,
                trt_opt_batch,
                trt_max_batch,
                trt_height_min,
                trt_height_opt,
                trt_height_max,
                trt_width_min,
                trt_width_opt,
                trt_width_max,
                trt_token_count_min,
                trt_token_count_opt,
                trt_token_count_max,
                force_rebuild,
                static_shapes,
                version,
                use_controlnet,
                trt_name
            ],
            outputs=[trt_result],
        )

        button_export_default_unet.click(
            export_unet_to_trt,
            inputs=[
                trt_min_batch,
                trt_opt_batch,
                trt_max_batch,
                trt_height_min,
                trt_height_opt,
                trt_height_max,
                trt_width_min,
                trt_width_opt,
                trt_width_max,
                trt_token_count_min,
                trt_token_count_opt,
                trt_token_count_max,
                force_rebuild,
                static_shapes,
                version,
                use_controlnet,
                trt_name
            ],
            outputs=[trt_result],
        )

        button_export_lora_unet.click(
            export_lora_to_trt,
            inputs=[trt_unet_dropdown, trt_unet_profile_dropdown, trt_lora_dropdown, trt_lora_force_rebuild, no_refit],
            outputs=[trt_result],
        ) # todo: refresh unet

        button_export_controlnet.click(
            export_controlnet_to_trt,
            inputs=[trt_controlunet_dropdown, trt_controlunet_profile_dropdown, trt_control_dropdown, trt_control_force_rebuild],
            outputs=[trt_result]
        )  # todo: refresh all

    return [(trt_interface, "TensorRT", "tensorrt")]
