"""ComfyUI nodes for model-scoped diffusion-model patches."""

from comfy_api.latest import io

from .patcher_helpers import patch_minimax_h3_cache_model


class UC_MiniMaxH3Cache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3Cache",
            display_name="MiniMax H3 Cache",
            category="advanced/model/patches",
            description=(
                "Applies an approximate block-stack residual cache to a cloned "
                "MiniMax H3 model without globally modifying the Core model class."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "reuse_threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Maximum accumulated relative feature change allowed for "
                        "reuse. Higher values skip more work and may reduce fidelity."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sampling progress at which cache reuse may begin.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.90,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sampling progress after which cache reuse stops.",
                ),
                io.Int.Input(
                    "max_steps",
                    default=2,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Maximum number of consecutive block-stack skips.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    tooltip=(
                        "auto keeps the residual with the model, cuda requires CUDA, "
                        "and cpu offloads the cached residual."
                    ),
                ),
                io.Boolean.Input(
                    "verbose",
                    default=False,
                    tooltip="Print per-step cache decisions and a final skip summary.",
                ),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_steps: int,
        device: str,
        verbose: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            patch_minimax_h3_cache_model(
                model=model,
                reuse_threshold=reuse_threshold,
                start_percent=start_percent,
                end_percent=end_percent,
                max_steps=max_steps,
                device=device,
                verbose=verbose,
            )
        )
