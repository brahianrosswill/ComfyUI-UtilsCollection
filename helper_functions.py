from typing import List, Union
from PIL import Image
from enum import Enum
import numpy as np
import re
import torch
import cv2
import math

def round_to_nearest(n, m):
    return int((n + (m / 2)) // m) * m


# Tensor to PIL
def simpletensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# PIL to Tensor
def simplepil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def hex_to_rgb(hex_str: str, default=(255, 255, 255)):
    hex_str = hex_str.lstrip('#')
    try:
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_str) == 8:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))
    except ValueError:
        pass
    return default

def math_diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)

def pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))

def blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0: k += 1
    return (k, k)

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045,
                   rgb / 12.92,
                   ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6/29)**3,
                        t ** (1/3),
                        t / (3 * (6/29)**2) + 4/29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)], axis=-1).astype(np.float32)

def dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)

def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

def occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    return (err > threshold).astype(np.float32)

def grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0: return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)

def auto_delta_e_threshold(delta_e: np.ndarray) -> float:
    p75 = float(np.percentile(delta_e, 75))
    p90 = float(np.percentile(delta_e, 90))
    spread = p90 - p75
    threshold = p75 + max(spread * 0.4, 3.0) if spread > 5.0 else p75 + max(spread * 0.6, 4.0)
    return float(np.clip(threshold, 4.0, 60.0))

def auto_occlusion_threshold(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    p85 = float(np.percentile(err, 85))
    p95 = float(np.percentile(err, 95))
    threshold = p95 + max((p95 - p85) * 0.5, 0.5)
    return float(np.clip(threshold, 1.0, 15.0))

def match_histogram(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Adjust the pixel values of a source image such that its histogram
    matches that of a target template image.
    Both source and template should be 2D numpy arrays (a single channel).
    """
    oldshape = source.shape
    source_flat = source.ravel()
    template_flat = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source_flat, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def match_image_properties(
    original_tensor: torch.Tensor,
    generated_tensor: torch.Tensor,
    overall_weight: float,
    color_weight: float,
    lighting_weight: float,
    texture_preservation: float,
    mask_tensor: torch.Tensor = None,
) -> torch.Tensor:

    batch_size = generated_tensor.size(0)
    out_tensors = []

    orig_batch = original_tensor.size(0)
    mask_batch = mask_tensor.size(0) if mask_tensor is not None else 0

    for i in range(batch_size):
        orig_i = i if i < orig_batch else 0

        orig_np = np.clip(255.0 * original_tensor[orig_i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        gen_np = np.clip(255.0 * generated_tensor[i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        mask_np = None
        if mask_tensor is not None:
            mask_i = i if i < mask_batch else 0
            # Extract mask, it might be (H, W) or (1, H, W) or (C, H, W)
            # Typically comfy masks are (H, W)
            m_t = mask_tensor[mask_i].cpu().numpy()
            if m_t.ndim > 2:
                m_t = m_t.squeeze()
            if m_t.shape != gen_np.shape[:2]:
                m_t = cv2.resize(m_t, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_np = m_t[:, :, np.newaxis] # (H, W, 1)

        orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB)

        out_lab = np.copy(gen_lab).astype(np.float32)
        gen_lab_f = gen_lab.astype(np.float32)

        # Detail extraction using Bilateral Filter to preserve edges
        # We extract details from the generated image to add them back later
        if texture_preservation > 0.0:
            # Apply bilateral filter to the L channel to get the "base" lighting without textures
            gen_l_base = cv2.bilateralFilter(gen_lab_f[:, :, 0], d=9, sigmaColor=75, sigmaSpace=75)
            # The difference is our high-frequency texture/edge detail
            gen_l_detail = gen_lab_f[:, :, 0] - gen_l_base

            # Apply bilateral filter to the original image L channel as well before matching
            orig_l_base = cv2.bilateralFilter(orig_lab[:, :, 0].astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

            # Match the base (textureless) lighting histograms
            l_trans_base = match_histogram(gen_l_base, orig_l_base)

            # Add the generated image's original texture back onto the matched base
            l_trans = l_trans_base + (gen_l_detail * texture_preservation)
        else:
            # Standard matching if texture preservation is 0
            l_trans = match_histogram(gen_lab[:, :, 0], orig_lab[:, :, 0])

        l_weight = lighting_weight * overall_weight
        out_lab[:, :, 0] = gen_lab_f[:, :, 0] * (1.0 - l_weight) + l_trans * l_weight

        # 2. Color (A and B channels)
        # If color_weight > 0, we match the A and B histograms
        a_trans = match_histogram(gen_lab[:, :, 1], orig_lab[:, :, 1])
        b_trans = match_histogram(gen_lab[:, :, 2], orig_lab[:, :, 2])
        c_weight = color_weight * overall_weight

        out_lab[:, :, 1] = gen_lab_f[:, :, 1] * (1.0 - c_weight) + a_trans * c_weight
        out_lab[:, :, 2] = gen_lab_f[:, :, 2] * (1.0 - c_weight) + b_trans * c_weight

        # Apply soft masking if provided
        if mask_np is not None:
            out_lab = gen_lab_f * (1.0 - mask_np) + out_lab * mask_np

        out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
        res_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

        out_tensor = torch.from_numpy(res_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0)

def composite(original_np: np.ndarray,
               generated_np: np.ndarray,
               delta_e_threshold: float,
               flow_preset: int,
               occlusion_threshold: float,
               grow_px: int,
               close_radius: int,
               min_region_px: int,
               feather_px: float) -> tuple:

    H, W = original_np.shape[:2]
    diag = math_diag(H, W)

    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    flow_fwd = dis_flow(gray_orig, gray_gen, flow_preset)
    flow_bwd = dis_flow(gray_gen, gray_orig, flow_preset)

    warped_gen_dense = warp(generated_np.astype(np.float32), flow_fwd)

    blur_kernel = blur_kernel_for_diag(diag)
    orig_blur = cv2.GaussianBlur(original_np, blur_kernel, 0)
    wgen_blur = cv2.GaussianBlur(warped_gen_dense, blur_kernel, 0)

    orig_lab = rgb_to_lab(orig_blur.reshape(-1, 3)).reshape(H, W, 3)
    wgen_lab = rgb_to_lab(wgen_blur.reshape(-1, 3)).reshape(H, W, 3)

    lab_diff = orig_lab - wgen_lab
    lab_diff[..., 0] *= 0.7
    delta_e = np.sqrt((lab_diff**2).sum(axis=2))

    sk = max(blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0: sk += 1
    delta_e_smooth = cv2.GaussianBlur(delta_e, (sk, sk), 0)

    auto_report = {}
    if delta_e_threshold < 0:
        delta_e_threshold = auto_delta_e_threshold(delta_e_smooth)
        auto_report["auto_delta_e"] = delta_e_threshold

    if occlusion_threshold < 0:
        occlusion_threshold = auto_occlusion_threshold(flow_fwd, flow_bwd)
        auto_report["auto_occlusion"] = occlusion_threshold

    occluded = occlusion_mask(flow_fwd, flow_bwd, occlusion_threshold)

    changed = np.maximum((delta_e_smooth > delta_e_threshold).astype(np.float32), occluded)

    if grow_px != 0:
        changed = grow_mask(changed, grow_px)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if min_region_px > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((changed > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_px:
                changed[labeled == i] = 0

    sharp_mask = changed.copy()

    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 3.0
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    y_grid, x_grid = np.mgrid[0:H:10, 0:W:10]
    pts_orig = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)

    flow_sub = flow_fwd[0:H:10, 0:W:10].reshape(-1, 2)
    mask_sub = sharp_mask[0:H:10, 0:W:10].reshape(-1)

    bg_idx = mask_sub < 0.1
    M = None
    if bg_idx.sum() > 10:
        src_pts = pts_orig[bg_idx]
        dst_pts = src_pts + flow_sub[bg_idx]

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is not None:
        final_aligned_gen = cv2.warpAffine(
            generated_np.astype(np.float32),
            M.astype(np.float64),
            (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        final_aligned_gen = generated_np

    m3 = composite_mask[..., np.newaxis]
    result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    flow_mag = np.sqrt((flow_fwd**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int(occluded.sum()),
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
    }
    stats.update(auto_report)

    return result, composite_mask, stats


def fill_mask_from_edges(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    inpaint_radius: int,
    edge_blend_blur: int,
) -> torch.Tensor:

    batch_size = image_tensor.size(0)
    out_tensors = []
    mask_batch = mask_tensor.size(0)

    for i in range(batch_size):
        # ComfyUI images are typically [B, H, W, C], float32, 0.0-1.0
        # Convert to numpy uint8 for OpenCV
        img_np = np.clip(255.0 * image_tensor[i].cpu().numpy(), 0, 255).astype(np.uint8)

        # If the image has an alpha channel, we only inpaint the RGB channels
        has_alpha = img_np.shape[-1] == 4
        if has_alpha:
            alpha_channel = img_np[:, :, 3]
            img_np = img_np[:, :, :3]

        # Extract and format the mask
        mask_i = i if i < mask_batch else 0
        m_t = mask_tensor[mask_i].cpu().numpy()

        if m_t.ndim > 2:
            m_t = m_t.squeeze()
        if m_t.shape[:2] != img_np.shape[:2]:
            m_t = cv2.resize(m_t, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        # OpenCV inpaint requires a strictly binary 8-bit mask
        mask_np = np.clip(255.0 * m_t, 0, 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # Apply Navier-Stokes Inpainting (pulls edge pixels inward)
        inpainted = cv2.inpaint(img_np, mask_binary, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

        # Smooth boundary blending
        if edge_blend_blur > 0:
            # Ensure blur kernel size is odd
            blur_size = edge_blend_blur if edge_blend_blur % 2 == 1 else edge_blend_blur + 1

            # Create a soft mask for alpha blending the inpainted result back onto the original
            soft_mask = cv2.GaussianBlur(mask_np.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
            soft_mask = soft_mask[:, :, np.newaxis]  # Reshape to [H, W, 1] for broadcasting

            img_f = img_np.astype(np.float32)
            inpainted_f = inpainted.astype(np.float32)

            # Composite: Original image where mask is 0, Inpainted image where mask is 1
            final_np = img_f * (1.0 - soft_mask) + inpainted_f * soft_mask
            final_np = np.clip(final_np, 0, 255).astype(np.uint8)
        else:
            final_np = inpainted

        # Restore alpha channel if it existed
        if has_alpha:
            final_np = np.dstack((final_np, alpha_channel))

        # Convert back to ComfyUI tensor [1, H, W, C]
        out_tensor = torch.from_numpy(final_np.astype(np.float32) / 255.0).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0)


def create_stretched_patch(img: np.ndarray, mask_binary: np.ndarray, axis: str, sample_thickness: int) -> np.ndarray:
    """
    Scans rows/cols to find the exact organic mask edges, samples a patch of 'sample_thickness',
    mirrors it, stretches it across the gap, and cross-fades.
    """
    H, W_img, C = img.shape
    out = np.copy(img).astype(np.float32)
    m_bool = (mask_binary > 127).astype(np.int8)

    if axis == 'horizontal':
        for y in range(H):
            row_mask = m_bool[y]
            if not np.any(row_mask): continue

            padded = np.pad(row_mask, (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            for start, end in zip(starts, ends):
                w = end - start + 1

                L_start = max(0, start - sample_thickness)
                L_width = start - L_start
                R_end = min(W_img, end + 1 + sample_thickness)
                R_width = R_end - (end + 1)

                L_stretch, R_stretch = None, None

                if L_width > 0:
                    L_strip = img[y, L_start:start].reshape(1, L_width, C)
                    L_strip = L_strip[:, ::-1, :] # Mirror horizontally
                    L_stretch = cv2.resize(L_strip, (w, 1)).reshape(w, C).astype(np.float32)

                if R_width > 0:
                    R_strip = img[y, end+1:R_end].reshape(1, R_width, C)
                    R_strip = R_strip[:, ::-1, :] # Mirror horizontally
                    R_stretch = cv2.resize(R_strip, (w, 1)).reshape(w, C).astype(np.float32)

                weights = np.linspace(1.0, 0.0, w).reshape(w, 1)

                if L_stretch is not None and R_stretch is not None:
                    out[y, start:end+1] = L_stretch * weights + R_stretch * (1.0 - weights)
                elif L_stretch is not None:
                    out[y, start:end+1] = L_stretch
                elif R_stretch is not None:
                    out[y, start:end+1] = R_stretch

    elif axis == 'vertical':
        for x in range(W_img):
            col_mask = m_bool[:, x]
            if not np.any(col_mask): continue

            padded = np.pad(col_mask, (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            for start, end in zip(starts, ends):
                h_seg = end - start + 1

                T_start = max(0, start - sample_thickness)
                T_height = start - T_start
                B_end = min(H, end + 1 + sample_thickness)
                B_height = B_end - (end + 1)

                T_stretch, B_stretch = None, None

                if T_height > 0:
                    T_strip = img[T_start:start, x].reshape(T_height, 1, C)
                    T_strip = T_strip[::-1, :, :] # Mirror vertically
                    T_stretch = cv2.resize(T_strip, (1, h_seg)).reshape(h_seg, C).astype(np.float32)

                if B_height > 0:
                    B_strip = img[end+1:B_end, x].reshape(B_height, 1, C)
                    B_strip = B_strip[::-1, :, :] # Mirror vertically
                    B_stretch = cv2.resize(B_strip, (1, h_seg)).reshape(h_seg, C).astype(np.float32)

                weights = np.linspace(1.0, 0.0, h_seg).reshape(h_seg, 1)

                if T_stretch is not None and B_stretch is not None:
                    out[start:end+1, x] = T_stretch * weights + B_stretch * (1.0 - weights)
                elif T_stretch is not None:
                    out[start:end+1, x] = T_stretch
                elif B_stretch is not None:
                    out[start:end+1, x] = B_stretch

    return out


def iterative_directional_stretch_fill(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    stretch_axis: str,
    sample_thickness: int,
    edge_blend_blur: int,
    iterations: int,
    mask_decay_pixels: int,
) -> torch.Tensor:

    batch_size = image_tensor.size(0)
    out_tensors = []
    mask_batch = mask_tensor.size(0)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i in range(batch_size):
        img_np = np.clip(255.0 * image_tensor[i].cpu().numpy(), 0, 255).astype(np.uint8)

        has_alpha = img_np.shape[-1] == 4
        if has_alpha:
            alpha_channel = img_np[:, :, 3]
            img_np = img_np[:, :, :3]

        mask_i = i if i < mask_batch else 0
        m_t = mask_tensor[mask_i].cpu().numpy()

        if m_t.ndim > 2:
            m_t = m_t.squeeze()
        if m_t.shape[:2] != img_np.shape[:2]:
            m_t = cv2.resize(m_t, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_np = np.clip(255.0 * m_t, 0, 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        working_img = np.copy(img_np)
        working_mask = np.copy(mask_binary)

        for step in range(iterations):
            if cv2.countNonZero(working_mask) == 0:
                break

            current_axis = stretch_axis
            if current_axis == 'auto':
                x, y, w, h = cv2.boundingRect(working_mask)
                current_axis = 'horizontal' if w < h else 'vertical'

            # Pass the full image and mask into our stretch function
            canvas = create_stretched_patch(working_img, working_mask, current_axis, sample_thickness)

            if edge_blend_blur > 0:
                blur_size = edge_blend_blur if edge_blend_blur % 2 == 1 else edge_blend_blur + 1
                soft_mask = cv2.GaussianBlur(working_mask.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
                soft_mask = soft_mask[:, :, np.newaxis]
            else:
                soft_mask = (working_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            working_f = working_img.astype(np.float32)
            merged = working_f * (1.0 - soft_mask) + canvas * soft_mask
            working_img = np.clip(merged, 0, 255).astype(np.uint8)

            if step < iterations - 1 and mask_decay_pixels > 0:
                working_mask = cv2.erode(working_mask, erosion_kernel, iterations=mask_decay_pixels)

        if has_alpha:
            working_img = np.dstack((working_img, alpha_channel))

        out_tensor = torch.from_numpy(working_img.astype(np.float32) / 255.0).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0)


def get_token_count(clip, text):
        """
        Robustly tokenizes a text segment and returns the number of its content tokens.
        """
        if not text:
            return 0

        tokens = clip.tokenize(text)

        max_content_len = 0
        for key in tokens:
            if len(tokens[key]) > 0 and len(tokens[key][0]) > 0:

                content_len = len(tokens[key][0]) - 2
                if content_len > max_content_len:
                    max_content_len = content_len

        return max(0, max_content_len)


def get_token_count_scaled(clip, text, **kwargs):
    """
    Robustly tokenizes a text segment and returns the number of its content tokens.
    Calculates true length by ignoring padding tokens added by fixed-length tokenizers (like Qwen3).
    """
    # Only return 0 if no text AND no llama_template (which adds tokens)
    if not text and not kwargs.get("llama_template"):
        return 0

    tokens = clip.tokenize(text, **kwargs)

    max_content_len = 0
    for key in tokens:
        if len(tokens[key]) > 0 and len(tokens[key][0]) > 0:
            token_list = tokens[key][0]

            # Count tokens that aren't padding.
            # In ComfyUI, padding tokens are usually 0 or the end-of-text token repeated.
            # We find the first occurrence of the end-of-text token or look for the padding.

            # For robust counting across models:
            # 1. Start with the full length
            # 2. Subtract 2 (for start and end tokens)
            # 3. If the length is still suspiciously large (like 510 or 254),
            #    it's likely a padded tokenizer. We try to find the actual content.

            raw_len = len(token_list)
            content_len = raw_len - 2

            # If it looks like a fixed-length padded result (common for T5/Qwen in ComfyUI)
            if raw_len >= 77:
                # Find the actual used length.
                # Most tokenizers pad with 0 or the last token (EOS).
                if isinstance(token_list, torch.Tensor):
                    ids = token_list.tolist()
                else:
                    ids = token_list

                # Qwen/Llama usually have a start token at 0.
                # Let's count how many IDs are actually distinct from the last token in the list
                # (which is usually the padding token)
                pad_id = ids[-1]
                actual_count = 0
                for i in range(1, len(ids) - 1): # Skip start token at 0 and end token
                    if ids[i] != pad_id:
                        actual_count += 1
                    else:
                        break # Hit padding
                content_len = actual_count

            if content_len > max_content_len:
                max_content_len = content_len

    return max(0, max_content_len)

def to_video_prompt(text: str, is_system: bool = False) -> str:
    """
    Transform image-based prompt presets into video-based ones by replacing
    static constraints with motion-focused directives.
    """
    # 1. Replacement Map for standard patterns (ordered by specificity)
    replacements = [
        # Complex Multi-phrase constraints
        (r"(?i)(?:Keep|Maintain|Ensure)\s+subject\s+position\s+and\s+their\s+pose\s+the\s+same\s+as\s+the\s+reference", "Maintain character consistency with natural lifelike motion"),
        (r"(?i)(?:Make\s+sure|Ensure)\s+the\s+subject\s+is\s+in\s+the\s+same\s+position", "Ensure the subject moves naturally within the environment"),
        (r"(?i)keeping\s+the\s+composition\s+and\s+structure\s+of\s+the\s+image\s+same\s+as\s+reference", "maintaining character consistency and cinematic flow"),
        (r"(?i)not\s+changing\s+the\s+positioning\s+of\s+subjects\s+in\s+image", "enabling dynamic character movement and large actions"),
        (r"(?i)strictly\s+maintaining\s+their\s+original\s+appearance", "preserving visual identity during dynamic motion"),

        # Simple Constraints -> Motion
        (r"(?i)Keep\s+pose", "Fluid movement and dynamic posing"),
        (r"(?i)Keep\s+angle", "Cinematic camera pans and motion"),
        (r"(?i)Keep\s+viewing\s+direction", "Dynamic gaze and perspective shifts"),
        (r"(?i)Keep\s+eyes", "Expressive eye movement and blinking"),
        (r"(?i)Keep\s+in\s+focus", "Maintain sharp cinematic focus"),
        (r"(?i)Keep\s+body\s+color", "Maintain consistent color during motion"),

        # Verb/Directive Transformation
        (r"(?i)Modify\s+(?:any\s+subjects'\s+appearance|the\s+scene)\s+to\s+(?:match\s+|look\s+like\s+|show\s+)?(.*?)(?:\.(?:\s|$)|$)", r"Animate with motion and physics authentic to \1. "),
        (r"(?i)focuses\s+on\s+edits\s+to\s+look\s+like\s+", "focuses on generating high-quality motion authentic to "),
        (r"(?i)modify\s+the\s+style\s+to\s+look\s+like\s+", "animate with motion and physics authentic to "),
        (r"(?i)image\s+editing\s+descriptions", "cinematic motion and video generation descriptions"),
        (r"(?i)keeping\s+the\s+structure\s+of\s+the\s+image\s+intact", "maintaining cinematic continuity"),

        # Nouns (Medium Swaps)
        (r"(?i)\bimage\b", "video"),
        (r"(?i)\bphotograph\b", "cinematic video"),
        (r"(?i)\bphotography\b", "cinematography"),
        (r"(?i)\bphoto\b", "video"),
        (r"(?i)\bstill\b", "video clip"),
        (r"(?i)\bscreenshot\b", "video clip"),
        (r"(?i)\bframe\b", "motion clip"),
        (r"(?i)\bcel\b", "animation frame"),
        (r"(?i)\bdrawing\b", "animation"),
        (r"(?i)\bpainting\b", "animated sequence"),
        (r"(?i)\billustration\b", "animated sequence"),
        (r"(?i)\bartwork\b", "animated sequence"),
    ]


    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    # Cleanup extra whitespace and broken punctuation
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\.\s*\.", ".", result)
    result = result.strip()

    if is_system:
        prefix = "Analyze cinematic information and dynamic potential. Describe action as if already in motion, focusing on large motions and style-appropriate physics. "
        result = prefix + result

    return result


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_H = "3:2 (Photo Format)"
    STANDARD_H = "4:3 (Standard Format)"
    CANVAS_H = "5:4 (Canvas Format)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"
    PANORAMA_H = "3:1 (Panorama)"
    PHOTO_V = "2:3 (Medium Portrait)"
    STANDARD_V = "3:4 (Standard Portrait)"
    CANVAS_V = "4:5 (Canvas Portrait)"
    WIDESCREEN_V = "9:16 (Tall Portrait)"
    PANORAMA_V = "1:3 (Tall Panorama)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.CANVAS_H: (5, 4),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
    AspectRatio.PANORAMA_H: (3, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.CANVAS_V: (4, 5),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.PANORAMA_V: (1, 3),
}

FLOW_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}
