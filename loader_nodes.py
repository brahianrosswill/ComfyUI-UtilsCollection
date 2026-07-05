import comfy
from folder_paths import get_filename_list, get_full_path_or_raise
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
from comfy_api.latest import io

_LORA_LOADER_CACHE = None


class UC_LoraLoaderCLIPOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraLoaderCLIPOnly",
            display_name="Load LoRA for CLIP Only",
            category="advanced/model",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("lora_name", get_filename_list("loras"), tooltip="Select a LoRA model to load. This node will attempt to extract and load only the CLIP portion of the LoRA, which can be useful for certain text/image embedding applications. Note that not all LoRA models will have a CLIP portion, and results may vary depending on the model architecture."),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="The strength of the CLIP portion to apply."),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, clip, lora_name: str, strength_clip: float) -> io.NodeOutput:
        global _LORA_LOADER_CACHE
        if strength_clip == 0:
            return (io.NodeOutput(clip),)
        # Placeholder for actual LoRA loading logic
        lora_path = get_full_path_or_raise("loras", lora_name)
        lora = None

        if _LORA_LOADER_CACHE is not None:
            if _LORA_LOADER_CACHE[0] == lora_path:
                lora = _LORA_LOADER_CACHE[1]
            else:
                _LORA_LOADER_CACHE = None

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            _LORA_LOADER_CACHE = (lora_path, lora)

        clip_lora = load_lora_for_models(None, clip, lora, 0, strength_clip)[1]
        return io.NodeOutput(clip_lora)

