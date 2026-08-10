from comfy_api.latest import io
from folder_paths import get_filename_list, get_full_path_or_raise

from .embedding_helpers import analyze_embedding_file


class UC_EmbeddingDetokenizerAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_EmbeddingDetokenizerAnalysis",
            display_name="Embedding Detokenizer Analysis",
            category="advanced/text",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "embedding_name",
                    options=get_filename_list("embeddings"),
                    tooltip="Select an embedding from ComfyUI's embeddings directory.",
                ),
                io.Int.Input(
                    "top_k",
                    default=5,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Number of nearest vocabulary tokens reported per embedding row.",
                ),
                io.Combo.Input(
                    "similarity_metric",
                    options=["cosine", "euclidean", "dot_product"],
                    default="cosine",
                ),
            ],
            outputs=[
                io.String.Output(display_name="analysis"),
                io.String.Output(display_name="approximation"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        embedding_name: str,
        top_k: int,
        similarity_metric: str,
    ) -> io.NodeOutput:
        embedding_path = get_full_path_or_raise("embeddings", embedding_name)
        analysis, approximation = analyze_embedding_file(
            clip,
            embedding_path,
            top_k,
            similarity_metric,
        )
        return io.NodeOutput(analysis, approximation)
