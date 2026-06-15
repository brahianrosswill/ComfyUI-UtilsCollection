
from enum import Enum
from comfy_api.latest import io
from comfy_extras.nodes_ideogram4 import Ideogram4Scheduler

class Ideogram4Enum(Enum):
    QUALITY = "Quality"
    HIGH = "High"
    DEFAULT = "Default"
    FAST = "Fast"
    TURBO = "Turbo"

IDEOGRAM4_PRESET_CONFIGS = {
  Ideogram4Enum.QUALITY.value: {
    "num_steps": 48,
    "mu": 0.0,
    "std": 1.5,
    "preset_id": "V4_QUALITY_48"
  },
  Ideogram4Enum.HIGH.value: {
    "num_steps": 34,
    "mu": 0.0,
    "std": 1.6875,
    "preset_id": "V4_HIGH_34"
  },
  Ideogram4Enum.DEFAULT.value: {
    "num_steps": 20,
    "mu": 0.0,
    "std": 1.75,
    "preset_id": "V4_DEFAULT_20"
  },
  Ideogram4Enum.FAST.value: {
    "num_steps": 16,
    "mu": 0.25,
    "std": 1.8375,
    "preset_id": "V4_FAST_16"
  },
  Ideogram4Enum.TURBO.value: {
    "num_steps": 12,
    "mu": 0.5,
    "std": 1.75,
    "preset_id": "V4_TURBO_12"
  }
}

class Ideogram4SchedulerPreset(Ideogram4Scheduler):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Ideogram4SchedulerPreset",
            display_name="Ideogram 4 Scheduler (Presets)",
            category="sampling/custom_sampling/schedulers",
            description="Schedule Presets for Ideogram 4. They are as follows: Quality=48, High=34, Default=20, Fast=16, Turbo=12",
            inputs=[
                io.Combo.Input("preset", options=[e.value for e in Ideogram4Enum], default=Ideogram4Enum.DEFAULT.value),
                io.Int.Input("width", default=1024, min=256, max=8192, step=16),
                io.Int.Input("height", default=1024, min=256, max=8192, step=16),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, preset, width, height) -> io.NodeOutput:
        config = IDEOGRAM4_PRESET_CONFIGS.get(preset)
        if not config:
            raise ValueError(f"Invalid preset: {preset}")

        return super().execute(
            steps=config["num_steps"],
            width=width,
            height=height,
            mu=config["mu"],
            std=config["std"]
        )


