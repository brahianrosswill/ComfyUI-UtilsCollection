import os
import numpy as np
import torch
import hashlib
import scipy
from tqdm import tqdm
import pilgram
from PIL import Image, ImageOps, ImageSequence, ImageDraw, ImageFont, ImageFilter
import kornia.morphology as morph
from .helper_functions import pil2tensor, tensor2pil, simplepil2tensor, simpletensor2pil, math_diag, pct_to_px, composite, fill_mask_from_edges, iterative_directional_stretch_fill, hex_to_rgb, match_image_properties, FLOW_PRESETS


from comfy_api.latest import io
from comfy import model_management
import node_helpers
from nodes import MAX_RESOLUTION

class UC_Image_Color_Noise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Image_Color_Noise",
            category="utils",
            inputs=[
                io.Int.Input("width", default=512, max=4096, min=64, step=1),
                io.Int.Input("height", default=512, max=4096, min=64, step=1),
                io.Float.Input("frequency", default=0.5, max=100.0, min=0.0, step=0.01),
                io.Float.Input(
                    "attenuation", default=0.5, max=100.0, min=0.0, step=0.01
                ),
                io.Combo.Input(
                    "noise_type",
                    options=["grey", "white", "red", "pink", "green", "blue", "mix"],
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[
                io.Image.Output(display_name="noise_image"),
            ],
        )

    @classmethod
    def execute(cls, width, height, frequency, attenuation, noise_type, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise_image = cls.generate_power_noise(
            width, height, frequency, attenuation, noise_type, generator
        )
        return io.NodeOutput(pil2tensor(noise_image))

    @classmethod
    def generate_power_noise(
        cls, width, height, frequency, attenuation, noise_type, generator
    ):
        def normalize_array(arr):
            return (255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))).astype(
                np.uint8
            )

        def white_noise(w, h, gen):
            return torch.rand(h, w, generator=gen).numpy()

        def grey_noise_texture(w, h, att, gen):
            return torch.normal(mean=0, std=att, size=(h, w), generator=gen).numpy()

        def fourier_noise(w, h, att, power_modifier, gen):
            noise = grey_noise_texture(w, h, att, gen)
            fy = np.fft.fftfreq(h)[:, np.newaxis]
            fx = np.fft.fftfreq(w)
            f = np.sqrt(fx**2 + fy**2)
            f[0, 0] = 1.0
            power_spectrum = f**power_modifier
            fft_noise = np.fft.fft2(noise)
            fft_modified = fft_noise * power_spectrum
            inv_fft = np.fft.ifft2(fft_modified)
            return np.real(inv_fft)

        noise_array = np.zeros((height, width, 3), dtype=np.uint8)
        zeros_channel = np.zeros((height, width), dtype=np.uint8)

        if noise_type == "grey":
            luma = normalize_array(
                grey_noise_texture(width, height, attenuation, generator)
            )
            noise_array = np.stack([luma, luma, luma], axis=-1)

        elif noise_type == "white":
            r = normalize_array(white_noise(width, height, generator))
            g = normalize_array(white_noise(width, height, generator))
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "red":
            r = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, zeros_channel, zeros_channel], axis=-1)

        elif noise_type == "green":
            g = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, g, zeros_channel], axis=-1)

        elif noise_type == "blue":
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, zeros_channel, b], axis=-1)

        elif noise_type == "pink":
            base_texture = fourier_noise(width, height, attenuation, -1.0, generator)
            r = normalize_array(base_texture)
            g = (r * 0.75).astype(np.uint8)
            b = (r * 0.85).astype(np.uint8)
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "mix":
            r = normalize_array(
                fourier_noise(width, height, attenuation, -1.0, generator)
            )  # Pink Frequency
            g = normalize_array(
                fourier_noise(width, height, attenuation, 0.5, generator)
            )  # Green Frequency
            b = normalize_array(
                fourier_noise(width, height, attenuation, 1.0, generator)
            )  # Blue Frequency
            noise_array = np.stack([r, g, b], axis=-1)

        else:
            print(f"[ERROR] Unsupported noise type `{noise_type}`")
            return Image.new("RGB", (width, height), color="black")

        return Image.fromarray(noise_array, "RGB")


class UC_LoadImagePath(io.ComfyNode):
    """
    Load an image from an arbitrary file path with proper mask handling.
    Returns the image and a mask extracted from the alpha channel.
    For images without alpha, returns a full-sized zero mask (not 64x64).
    Supports both absolute and relative paths, with any OS path separator.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Normalize a file path to handle:
        - Backslashes (Windows) and forward slashes (Unix)
        - Relative paths (starting with '.', '..', or lowercase letter)
        - Whitespaces in paths and filenames

        Returns an absolute, normalized path.
        """
        if not path:
            return path

        # Strip leading/trailing whitespace but preserve internal whitespace
        path = path.strip()

        # Normalize path separators: replace backslashes with forward slashes
        # Then use os.path.normpath to get the OS-appropriate format
        path = path.replace('\\', '/')
        path = os.path.normpath(path)

        # Convert to absolute path if relative
        # os.path.abspath handles: '.', '..', and paths without drive letter
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadImagePath",
            display_name="Load Image (Path)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "image_path",
                    multiline=False,
                    placeholder="path/to/image.png or X:/path/to/image.png",
                    tooltip="Path to the image file. Supports absolute or relative paths with any OS format (backslashes or forward slashes). Whitespaces in paths are supported.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Mask.Output(display_name="MASK"),
                io.Mask.Output(display_name="MASK_INVERTED"),
            ],
        )

    @classmethod
    def execute(cls, image_path: str) -> io.NodeOutput:
        # Normalize the path to handle relative paths and different separators
        normalized_path = cls._normalize_path(image_path)

        # Validate path
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(f"Invalid image path: {image_path} (resolved to: {normalized_path})")

        img = node_helpers.pillow(Image.open, normalized_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # Handle 16-bit images (mode 'I') - normalize by 65535, not 255
            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            # Set dimensions from first frame
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # Skip frames with different dimensions
            if image.size[0] != w or image.size[1] != h:
                continue

            # Convert to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            # Extract mask from alpha channel
            if 'A' in i.getbands():
                # RGBA image - extract alpha
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            elif i.mode == 'P' and 'transparency' in i.info:
                # Palette mode with transparency - convert to RGBA (already transposed)
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            else:
                # No alpha - return full-sized zero mask (NOT 64x64!)
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            # MPO format: only use first frame
            if img.format == "MPO":
                break

        if len(output_images) == 0:
            raise ValueError(f"No valid image frames could be loaded from: {image_path}")

        # Stack frames
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Create inverted mask
        output_mask_inverted = 1.0 - output_mask

        return io.NodeOutput(output_image, output_mask, output_mask_inverted)

    @classmethod
    def IS_CHANGED(cls, image_path: str):
        normalized_path = cls._normalize_path(image_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        m = hashlib.sha256()
        with open(normalized_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image_path: str):
        if not image_path:
            return "Image path cannot be empty"
        normalized_path = cls._normalize_path(image_path)
        if not os.path.isfile(normalized_path):
            return f"Invalid image file: {image_path} (resolved to: {normalized_path})"
        return True


class UC_LoadImageDirectory(io.ComfyNode):
    """
    Load multiple images from a directory as a batch.
    Supports selecting a range of images using start index and count.
    Images are sorted alphanumerically.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return path
        path = path.strip()
        path = path.replace('\\', '/')
        path = os.path.normpath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadImageDirectory",
            display_name="Load Images (Directory)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "directory_path",
                    multiline=False,
                    placeholder="path/to/directory",
                    tooltip="Path to the directory containing images.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Index of the first image to load (sorted alphabetically)."
                ),
                io.Int.Input(
                    "load_count",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of images to load."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE", is_output_list=True),
                io.Mask.Output(display_name="MASK", is_output_list=True),
                io.Mask.Output(display_name="MASK_INVERTED", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, directory_path: str, start_index: int, load_count: int) -> io.NodeOutput:
        normalized_path = cls._normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        # Get valid image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        # Apply slice
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        if not selected_files:
             raise ValueError(f"No images found in range [{start_index}:{end_index}] in directory: {directory_path}")

        # Group images by size to handle batching or provide warning
        output_images = []
        output_masks = []
        output_masks_inverted = []

        for file_path in selected_files:
            w, h = None, None
            try:
                img = node_helpers.pillow(Image.open, file_path)
            except Exception as e:
                print(f"Warning: Could not load image {file_path}: {e}")
                continue

            # Process just the first frame
            i = node_helpers.pillow(ImageOps.exif_transpose, img)

            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            if w is None:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                print(f"Warning: Skipping {file_path} due to dimension mismatch. Expected {w}x{h}, got {image.size}")
                continue

            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            elif i.mode == 'P' and 'transparency' in i.info:
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            else:
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                mask_inverted = 1.0 - mask


            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))
            output_masks_inverted.append(mask_inverted.unsqueeze(0))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")


        return io.NodeOutput(output_images, output_masks, output_masks_inverted)

    @classmethod
    def IS_CHANGED(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = cls._normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                # Hash filename and mtime
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()


class UC_ImageMatchPropertiesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageMatchProperties",
            display_name="Image Match Properties",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Float.Input("overall_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("color_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("lighting_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("texture_preservation", default=0.5, min=0.0, max=1.0, step=0.001, tooltip="Preserves edges and textures from the generated image by matching only low-frequency properties."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask to softly blend the color/lighting changes onto the generated image."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        overall_weight: float,
        color_weight: float,
        lighting_weight: float,
        texture_preservation: float,
        mask: torch.Tensor = None,
    ) -> io.NodeOutput:
        result = match_image_properties(
            original_image,
            generated_image,
            overall_weight,
            color_weight,
            lighting_weight,
            texture_preservation,
            mask,
        )
        return io.NodeOutput(result)


class UC_OpticalFlowComposite(io.ComfyNode):
    """
    Composites a Klein edit onto the original image.

    v2.2: Global Rigid Alignment. Calculates a single global camera shift from
    unchanged background pixels and translates the entire generated image rigidly.
    Eliminates seam distortion while fixing AI background drift.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_OpticalFlowComposite",
            display_name="Optical Flow Composite (Global Align)",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Float.Input(
                    "delta_e_threshold",
                    default=-1.0, min=-1.0, max=100.0, step=1.0,
                    tooltip="How different a pixel's color must be to count as 'edited'. Higher values = only obvious edits are detected (smaller mask, more original preserved). Lower values = subtle changes are also captured (larger mask, more of the generated image used). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "grow_mask_pct",
                    default=0.0, min=-3.0, max=3.0, step=0.1,
                    tooltip="Expands or shrinks the detected edit region. Positive values grow the mask outward, capturing more of the surrounding area (useful if edges of the edit are being clipped). Negative values erode the mask inward, trimming the edges (useful if too much background is being pulled in)."
                ),
                io.Float.Input(
                    "feather_pct",
                    default=2.0, min=0.0, max=10.0, step=0.25,
                    tooltip="How gradually the edit blends into the original at the mask boundary. Higher values create a wider, softer transition (smoother blending, but may wash out fine edges). Lower values create a sharper, more abrupt cutover (crisper edges, but seams may be more visible)."
                ),
                io.Combo.Input(
                    "flow_quality",
                    options=["medium", "fast", "ultrafast"],
                    default="medium",
                    tooltip="Accuracy of the optical flow alignment between original and generated images. Higher quality = more precise change detection and alignment (slower). Lower quality = faster processing but may miss subtle shifts or produce noisier masks."
                ),
                io.Float.Input(
                    "occlusion_threshold",
                    default=-1.0, min=-1.0, max=20.0, step=0.5,
                    tooltip="Sensitivity to pixels that moved so much they can't be reliably matched between images. Higher values ignore more motion discrepancies (fewer false positives from camera jitter, but may miss real edits). Lower values flag more pixels as changed (catches more edits, but may over-detect in noisy areas). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "close_radius_pct",
                    default=0.5, min=0.0, max=5.0, step=0.1,
                    tooltip="Fills small holes and gaps inside the detected edit region. Higher values close larger gaps (creates a more solid, continuous mask). Lower values leave small holes intact (preserves finer mask detail but may leave speckled artifacts inside the edit)."
                ),
                io.Float.Input(
                    "min_region_pct",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Removes small isolated blobs from the mask that are likely false positives. Higher values filter out larger stray regions (cleaner mask, but may discard small intentional edits). Lower values keep smaller regions (preserves tiny edits, but may let through noise)."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="composited_image"),
                io.Mask.Output(display_name="change_mask"),
                io.String.Output(display_name="report"),
            ]
        )

    @classmethod
    def execute(cls, original_image, generated_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, min_region_pct=0.05):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W = gen_np.shape[:2]
            pil  = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W = gen_np.shape[:2]
        diag = math_diag(H, W)
        total_area = H * W

        grow_px    = round(grow_mask_pct * diag / 100.0)
        feather_px = abs(feather_pct) * diag / 100.0
        close_px   = pct_to_px(close_radius_pct, diag)
        min_px     = max(0, round(min_region_pct * total_area / 100.0))

        result, change_mask, stats = composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            min_region_px       = min_px,
            feather_px          = feather_px,
        )

        report_lines =[
            "=== Klein Edit Composite v2.2 (Global Align) ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            f"",
        ]

        if "auto_delta_e" in stats:
            report_lines.append(f"ΔE threshold:     AUTO → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"ΔE threshold:     {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion thresh: AUTO → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion thresh: {occlusion_threshold:.1f}")

        report_lines +=[
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Close radius:     {close_radius_pct:.1f}% → {close_px}px",
            f"Min region:       {min_region_pct:.2f}% → {min_px}px",
            f"Flow quality:     {flow_quality}",
            f"",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Occluded pixels:  {stats['occluded_px']:,}",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
            f"Flow p99 shift:   {stats['flow_p99_px']:.2f}px",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]

        return io.NodeOutput(torch.from_numpy(result).unsqueeze(0),
                torch.from_numpy(change_mask).unsqueeze(0),
                "\n".join(report_lines))


class UC_ImageInwardEdgeFill(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageInwardEdgeFill",
            display_name="Image Inward Edge Fill",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input(
                    "inpaint_radius",
                    default=3,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="How far the algorithm looks for edge pixels. Higher values are slower but better for large holes."
                ),
                io.Int.Input(
                    "edge_blend_blur",
                    default=9,
                    min=0,
                    max=101,
                    step=2,
                    tooltip="Applies a Gaussian blur to the mask to smoothly blend the filled area with the original edges. 0 disables it."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        inpaint_radius: int,
        edge_blend_blur: int,
    ) -> io.NodeOutput:
        result = fill_mask_from_edges(
            image,
            mask,
            inpaint_radius,
            edge_blend_blur,
        )
        return io.NodeOutput(result)


class UC_ImageIterativeStretchFill(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageIterativeStretchFill",
            display_name="Image Iterative Stretch Fill",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "stretch_axis",
                    default="auto",
                    options=["auto", "horizontal", "vertical"],
                    tooltip="'Auto' stretches across the narrowest dimension of the current mask."
                ),
                io.Int.Input(
                    "sample_thickness",
                    default=32,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="How many pixels of unmasked image to grab from the edges to stretch inwards."
                ),
                io.Int.Input(
                    "edge_blend_blur",
                    default=9,
                    min=0,
                    max=101,
                    step=2,
                    tooltip="Softens the mask boundary to seamlessly blend the stretched fill."
                ),
                io.Int.Input(
                    "iterations",
                    default=5,
                    min=1,
                    max=50,
                    step=1,
                    tooltip="Number of times to repeat the stretch and fill process."
                ),
                io.Int.Input(
                    "mask_decay_pixels",
                    default=4,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Shrinks the mask by this many pixels each iteration, creating a telescoping fill effect."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        stretch_axis: str,
        sample_thickness: int,
        edge_blend_blur: int,
        iterations: int,
        mask_decay_pixels: int,
    ) -> io.NodeOutput:
        result = iterative_directional_stretch_fill(
            image,
            mask,
            stretch_axis,
            sample_thickness,
            edge_blend_blur,
            iterations,
            mask_decay_pixels,
        )
        return io.NodeOutput(result)


class UC_TextOverlayNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TextOverlayNode",
            display_name="Text Overlay",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("text", multiline=True, default="Hello World"),
                io.Int.Input("font_size", default=32, min=1, max=1024),
                io.String.Input("text_color", default="FFFFFF"),
                io.String.Input("bg_color", default="000000"),
                io.Boolean.Input("draw_background", default=True),
                io.Int.Input("bg_padding", default=10, min=0, max=1024),
                io.Float.Input("bg_transparency", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="0.0 is fully transparent, 1.0 is fully opaque"),
                io.Boolean.Input("use_percentage", default=False, tooltip="If True, top/bottom/left/right are treated as percentages (0-100) of the image size."),
                io.Int.Input("top", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use bottom offset"),
                io.Int.Input("bottom", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use top offset"),
                io.Int.Input("left", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use right offset"),
                io.Int.Input("right", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use left offset"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        text: str,
        font_size: int,
        text_color: str,
        bg_color: str,
        draw_background: bool,
        bg_padding: int,
        bg_transparency: float,
        use_percentage: bool,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> io.NodeOutput:

        t_color = hex_to_rgb(text_color, (255, 255, 255))

        # Calculate background color with transparency (alpha 0-255)
        b_color_base = hex_to_rgb(bg_color, (0, 0, 0))
        alpha = int(bg_transparency * 255.0)
        # Ensure b_color is exactly 4 elements long for RGBA
        if len(b_color_base) == 3:
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)
        else: # Handle case where hex_to_rgb returns a 4-element tuple or default
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Linux fallback
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        # Handle batch of images
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        output_images = []

        for i in range(batch_count):
            img_tensor = image[i] if batch_count > 1 else image
            # Tensor is typically (C, H, W) or (H, W, C) depending on context, assuming (H, W, C) here
            # Convert to PIL
            img_pil = Image.fromarray(np.clip(255.0 * img_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGBA")

            draw = ImageDraw.Draw(img_pil)

            # Calculate text size using textbbox
            left_box, top_box, right_box, bottom_box = draw.textbbox((0, 0), text, font=font)
            text_width = right_box - left_box
            text_height = bottom_box - top_box

            img_width, img_height = img_pil.size

            # Calculate total width/height including background padding
            total_width = text_width + (bg_padding * 2 if draw_background else 0)
            total_height = text_height + (bg_padding * 2 if draw_background else 0)

            # Resolve coordinates based on mode (pixels vs percentage)
            def resolve_coord(val, max_val):
                if val == -1:
                    return -1
                if use_percentage:
                    return int((val / 100.0) * max_val)
                return val

            l_resolved = resolve_coord(left, img_width)
            r_resolved = resolve_coord(right, img_width)
            t_resolved = resolve_coord(top, img_height)
            b_resolved = resolve_coord(bottom, img_height)

            # Determine X position
            if l_resolved == -1 and r_resolved == -1:
                x_pos = (img_width - total_width) // 2
            elif l_resolved != -1:
                x_pos = l_resolved
            else: # r_resolved != -1
                x_pos = img_width - total_width - r_resolved

            # Determine Y position
            if t_resolved == -1 and b_resolved == -1:
                y_pos = (img_height - total_height) // 2
            elif t_resolved != -1:
                y_pos = t_resolved
            else: # b_resolved != -1
                y_pos = img_height - total_height - b_resolved

            # Draw background
            if draw_background:
                bg_rect = [x_pos, y_pos, x_pos + total_width, y_pos + total_height]
                # To support alpha, we draw on a separate layer and composite
                overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=b_color)
                img_pil = Image.alpha_composite(img_pil, overlay)
                draw = ImageDraw.Draw(img_pil) # Re-init draw for text on composited image

            # Draw text
            text_x = x_pos + (bg_padding if draw_background else 0)
            text_y = y_pos + (bg_padding if draw_background else 0)

            # Use textbbox offset for more accurate vertical alignment of text
            draw.text((text_x - left_box, text_y - top_box), text, fill=t_color, font=font)

            # Convert back to tensor (RGB)
            img_pil = img_pil.convert("RGB")
            out_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)
            output_images.append(out_tensor)

        if batch_count > 1:
            out = torch.stack(output_images, dim=0)
        else:
            out = output_images[0].unsqueeze(0)

        return io.NodeOutput(out)


class UC_ModifyMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ModifyMask",
            category="utils/mask",
            display_name="Modify Mask (Expand/Contract with options)",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "expand", default=0, max=MAX_RESOLUTION, min=-MAX_RESOLUTION, step=1
                ),
                io.Float.Input(
                    "incremental_expandrate", default=0.0, max=100.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("tapered_corners", default=True),
                io.Boolean.Input("flip_input", default=False),
                io.Float.Input(
                    "blur_radius", default=0.0, max=100.0, min=0.0, step=0.01
                ),
                io.Float.Input("lerp_alpha", default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input(
                    "decay_factor", default=1.0, max=1.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("fill_holes", default=False, optional=True),
                io.Float.Input("lower_clamp", default=0.0, max=100.0, min=0.0, step=0.1),
                io.Float.Input("upper_clamp", default=100.0, max=100.0, min=0.0, step=0.1),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="mask_inverted"),
            ],
        )

    @classmethod
    def execute(
        self,
        mask,
        expand,
        tapered_corners,
        flip_input,
        blur_radius,
        incremental_expandrate,
        lerp_alpha,
        decay_factor,
        fill_holes=False,
        lower_clamp=0.0,
        upper_clamp=100.0,
    ):

        alpha = lerp_alpha
        decay = decay_factor

        # 1. Clone the original mask to keep a reference to the un-blurred pixels
        original_mask_input = mask.clone()

        if flip_input:
            mask = 1.0 - mask
            original_mask_input = 1.0 - original_mask_input

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

        # Prepare original mask for processing loop (match dimensions)
        original_mask_batches = original_mask_input.reshape(
            (-1, mask.shape[-2], mask.shape[-1])
        )

        out = []
        previous_output = None
        current_expand = expand
        for m in tqdm(growmask, desc="Expanding/Contracting Mask"):
            output = (
                m.unsqueeze(0).unsqueeze(0).to(model_management.get_torch_device())
            )  # Add batch and channel dims for kornia
            if abs(round(current_expand)) > 0:
                # Create kernel - kornia expects kernel on same device as input
                if tapered_corners:
                    kernel = torch.tensor(
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        dtype=torch.float32,
                        device=output.device,
                    )
                else:
                    kernel = torch.tensor(
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=torch.float32,
                        device=output.device,
                    )

                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = morph.erosion(output, kernel)
                    else:
                        output = morph.dilation(output, kernel)

            output = output.squeeze(0).squeeze(0)  # Remove batch and channel dims

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)

            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = scipy.ndimage.binary_fill_holes(output_np)
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)

            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output.cpu())

        if blur_radius != 0 and current_expand != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                blurred_tensor = pil2tensor(pil_image)

                # 2. Restore the original pixels IF we are expanding
                # We use torch.max: this keeps the original pixel value unless the
                # blurred expansion is brighter. It prevents "adding" values together.
                if current_expand > 0:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.max(blurred_tensor, original_slice)
                else:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.min(blurred_tensor, original_slice)

                out[idx] = blurred_tensor

            blurred = torch.cat(out, dim=0)
            if lower_clamp > 0.0:
                blurred = torch.max(blurred, torch.tensor(lower_clamp / 100.0, device=blurred.device))
            if upper_clamp < 100.0:
                blurred = torch.min(blurred, torch.tensor(upper_clamp / 100.0, device=blurred.device))
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        elif blur_radius != 0 and current_expand == 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            if lower_clamp > 0.0:
                blurred = torch.max(blurred, torch.tensor(lower_clamp / 100.0, device=blurred.device))
            if upper_clamp < 100.0:
                blurred = torch.min(blurred, torch.tensor(upper_clamp / 100.0, device=blurred.device))
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        else:
            mask = torch.stack(out, dim=0)
            if lower_clamp > 0.0:
                mask = torch.max(mask, torch.tensor(lower_clamp / 100.0, device=mask.device))
            if upper_clamp < 100.0:
                mask = torch.min(mask, torch.tensor(upper_clamp / 100.0, device=mask.device))
            mask_inverted = 1.0 - mask
            return io.NodeOutput(mask, mask_inverted)


class UC_ImageBlendByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageBlendByMask",
            category="utils/mask",
            display_name="Image Blend by Mask",
            inputs=[
                io.Image.Input("destination"),
                io.Image.Input("source"),
                io.Combo.Input(
                    "mode",
                    options=[
                        "add",
                        "color",
                        "color_burn",
                        "color_dodge",
                        "darken",
                        "difference",
                        "exclusion",
                        "hard_light",
                        "hue",
                        "lighten",
                        "multiply",
                        "overlay",
                        "screen",
                        "soft_light",
                    ],
                    default="add",
                ),
                io.Float.Input(
                    "blend_percentage", default=1.0, max=1.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("resize_source", default=False),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Image.Output(display_name="blended_image"),
            ],
        )

    @classmethod
    def execute(
        self,
        destination,
        source,
        mode="add",
        blend_percentage=1.0,
        resize_source=False,
        mask=None,
    ):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        source = source.movedim(-1, 1).to(destination.device)

        if resize_source:
            source = torch.nn.functional.interpolate(
                source,
                size=(destination.shape[-2], destination.shape[-1]),
                mode="bicubic",
            )

        # Convert images to PIL
        img_a = simpletensor2pil(destination)
        img_b = simpletensor2pil(source)

        # Apply blending mode
        blending_modes = {
            "color": pilgram.css.blending.color,
            "color_burn": pilgram.css.blending.color_burn,
            "color_dodge": pilgram.css.blending.color_dodge,
            "darken": pilgram.css.blending.darken,
            "difference": pilgram.css.blending.difference,
            "exclusion": pilgram.css.blending.exclusion,
            "hard_light": pilgram.css.blending.hard_light,
            "hue": pilgram.css.blending.hue,
            "lighten": pilgram.css.blending.lighten,
            "multiply": pilgram.css.blending.multiply,
            "add": pilgram.css.blending.normal,
            "overlay": pilgram.css.blending.overlay,
            "screen": pilgram.css.blending.screen,
            "soft_light": pilgram.css.blending.soft_light,
        }

        out_image = blending_modes.get(mode, pilgram.css.blending.normal)(img_a, img_b)

        out_image = out_image.convert("RGB")

        # Apply mask if provided
        if mask is not None:
            mask = ImageOps.invert(simpletensor2pil(mask).convert("L"))
            out_image = Image.composite(img_a, out_image, mask.resize(img_a.size))

        # Blend image based on blend percentage
        blend_mask = Image.new(
            mode="L", size=img_a.size, color=(round(blend_percentage * 255))
        )
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        blended_image = simplepil2tensor(out_image)
        return io.NodeOutput(blended_image)
