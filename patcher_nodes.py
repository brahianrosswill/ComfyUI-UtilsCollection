"""ComfyUI nodes for model-scoped diffusion-model patches."""

from comfy_api.latest import io

from .patcher_helpers import (
    SpectrumH3Config,
    effective_bootstrap_first_forecast,
    patch_minimax_h3_cache_model,
    patch_minimax_h3_spectrum_model,
)


class UC_MiniMaxH3Cache(io.ComfyNode):
    """Apply the MiniMax H3 whole-block-stack cache to a cloned model patcher.

    Implementation invariant: current Core exposes per-block replacements but no
    replacement boundary around the complete H3 block stack. The local ``_forward``
    adaptation deliberately adds that missing boundary through
    ``ModelPatcher.add_object_patch``. Core installs it only while the output clone
    is patched and restores the original bound method when unpatching; this is not
    a global class monkey-patch. Replacing it with chained per-block wrappers is not
    equivalent unless preservation of existing replacements, prefetch behavior,
    whole-stack residual capture, and ModelPatcher lifecycle is proven end to end.
    """

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


class UC_MiniMaxH3Spectrum(io.ComfyNode):
    """Apply experimental spectral feature forecasting to a cloned H3 model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3Spectrum",
            display_name="MiniMax H3 Spectrum (Experimental)",
            category="advanced/model/patches",
            description=(
                "Forecasts complete post-transformer MiniMax H3 features with a "
                "Chebyshev ridge model while retaining native output reconstruction."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "execution_mode",
                    options=["spectrum", "forced_actual"],
                    default="spectrum",
                    tooltip="Spectrum forecasts eligible steps; forced_actual measures patch overhead without forecasts.",
                ),
                io.Int.Input("degree", default=1, min=1, max=16, step=1, tooltip="Chebyshev polynomial degree. Values other than 1 disable first-forecast bootstrap."),
                io.Float.Input("ridge_lambda", default=0.1, min=0.0, max=10.0, step=0.01, tooltip="Regularization that stabilizes the history fit."),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.05, tooltip="Initial interval between actual transformer evaluations."),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=8.0, step=0.05, tooltip="Amount added to the actual-evaluation interval after each refresh."),
                io.Int.Input("warmup_steps", default=1, min=0, max=64, step=1, tooltip="Initial solver steps that always run the transformer. Values above 1 disable first-forecast bootstrap."),
                io.Int.Input("tail_actual_steps", default=1, min=0, max=64, step=1, tooltip="Final solver steps that always run the transformer."),
                io.Int.Input("max_history", default=8, min=2, max=64, step=1, tooltip="Maximum actual post-transformer features retained for fitting."),
                io.Float.Input("video_blend_weight", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Video spectral share; 1 uses pure spectral prediction and 0 uses local linear prediction."),
                io.Float.Input("audio_blend_weight", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Audio spectral share; default 0 avoids direct spectral mixing of audio rows."),
                io.Combo.Input("history_storage", options=["system_ram", "vram"], default="system_ram", tooltip="Device used for bounded causal feature history."),
                io.Boolean.Input("bootstrap_first_forecast", default=True, tooltip="Forecast step 1 from step 0 only when degree is 1 and warmup is at most 1."),
                io.Boolean.Input("offline_smoothing_replay", default=True, tooltip="Run optional two-pass smoothing replay to protect H3 audio continuity."),
                io.Combo.Input("offline_archive_storage", options=["system_ram", "vram"], default="system_ram", tooltip="Device used for replay anchors retained through both passes."),
                io.Boolean.Input("debug", default=False, tooltip="Log forecast decisions, timing, retained memory, and run summary."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        execution_mode: str,
        degree: int,
        ridge_lambda: float,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        tail_actual_steps: int,
        max_history: int,
        video_blend_weight: float,
        audio_blend_weight: float,
        history_storage: str,
        bootstrap_first_forecast: bool,
        offline_smoothing_replay: bool,
        offline_archive_storage: str,
        debug: bool,
    ) -> io.NodeOutput:
        config = SpectrumH3Config(
            enabled=True,
            force_actual=execution_mode == "forced_actual",
            degree=degree,
            ridge_lambda=ridge_lambda,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            max_history=max_history,
            blend_weight=video_blend_weight,
            audio_blend_weight=audio_blend_weight,
            history_storage=history_storage,
            bootstrap_first_forecast=effective_bootstrap_first_forecast(
                bootstrap_first_forecast, degree, warmup_steps
            ),
            offline_smoothing_replay=(
                offline_smoothing_replay if execution_mode == "spectrum" else False
            ),
            offline_archive_storage=offline_archive_storage,
            debug=debug,
        )
        return io.NodeOutput(patch_minimax_h3_spectrum_model(model, config))
