from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterator, Sequence

import av
import numpy as np
import torch

from .helper_functions import resize_nchw


VIDEO_FRAME_SAMPLING_STRATEGIES = (
    "codec keyframes",
    "uniform PTS",
    "focused PTS",
)

VIDEO_FRAME_TIMESTAMP_FORMATS = (
    "HH:MM:SS.mmm",
    "HH:MM:SS:mmm",
    "MM:SS.mmm",
    "MM:SS:mmm",
    "00.000s",
    "0.0s",
    "0.00s",
)

VIDEO_FRAME_TIMELINE_STYLES = (
    "H3 alignment prefix",
    "H3 pictures",
    "indexed",
    "timestamps only",
)

_VIDEO_TIMESTAMP_COLON_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+)(?P<fraction>[.:]\d+)?$"
)


def parse_video_timestamp(value) -> Fraction:
    """Parse one supported video timestamp into exact nonnegative seconds."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not timestamps")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        result = Fraction(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is empty")
        suffix = re.fullmatch(r"(.+?)\s*(?:s|seconds?)", text, re.IGNORECASE)
        if suffix:
            text = suffix.group(1).strip()
        colon_parts = text.split(":")
        if len(colon_parts) in (3, 4) and all(part.isdigit() for part in colon_parts):
            if len(colon_parts) == 4:
                hours, minutes, seconds, milliseconds = map(int, colon_parts)
            else:
                hours = 0
                minutes, seconds, milliseconds = map(int, colon_parts)
            if minutes >= 60 and hours:
                raise ValueError("minute component must be below 60")
            if seconds >= 60:
                raise ValueError("second component must be below 60")
            result = Fraction(hours * 3600 + minutes * 60 + seconds) + Fraction(milliseconds, 10 ** len(colon_parts[-1]))
            if result < 0:
                raise ValueError("timestamp must not be negative")
            return result
        match = _VIDEO_TIMESTAMP_COLON_PATTERN.fullmatch(text)
        if match:
            hours = int(match.group("hours") or 0)
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
            if minutes >= 60 and match.group("hours") is not None:
                raise ValueError("minute component must be below 60")
            if seconds >= 60:
                raise ValueError("second component must be below 60")
            fraction = match.group("fraction")
            fractional = Fraction(0)
            if fraction:
                digits = fraction[1:]
                fractional = Fraction(int(digits), 10 ** len(digits))
            result = Fraction(hours * 3600 + minutes * 60 + seconds) + fractional
        else:
            try:
                result = Fraction(text)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"unsupported timestamp {value!r}") from exc
    else:
        raise ValueError(f"unsupported timestamp type {type(value).__name__}")
    if result < 0:
        raise ValueError("timestamp must not be negative")
    return result


def parse_video_timestamps(value) -> list[Fraction]:
    """Flatten and parse timestamp containers while preserving source order."""
    raw = []

    def collect(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, str) and re.search(r"[,;\n]", item):
            parts = re.split(r"[,;\n]", item)
            if any(not part.strip() for part in parts):
                raise ValueError("timestamp list contains an empty item")
            raw.extend(parts)
        else:
            raw.append(item)

    collect(value)
    if not raw:
        raise ValueError("at least one timestamp is required")
    parsed = []
    for index, item in enumerate(raw, start=1):
        try:
            parsed.append(parse_video_timestamp(item))
        except ValueError as exc:
            raise ValueError(f"timestamp {index}: {exc}") from exc
    for index in range(1, len(parsed)):
        if parsed[index] < parsed[index - 1]:
            raise ValueError(f"timestamp {index + 1} is earlier than timestamp {index}")
    return parsed


@dataclass(frozen=True)
class VideoFrameRecord:
    """Presentation metadata for one frame in the active VIDEO input."""

    frame_index: int
    timestamp: Fraction
    key_frame: bool


@dataclass(frozen=True)
class SampledVideoFrames:
    image_batch: torch.Tensor
    image_list: list[torch.Tensor]
    timestamps: list[str]
    timestamps_text: str
    timeline_text: str
    video_runtime: float
    structured_timeline_text: str


def _as_fraction(value: Fraction | float | int) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _video_source_factory(video) -> Callable[[], str | io.BytesIO]:
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        return lambda: path

    seek = getattr(source, "seek", None)
    if callable(seek):
        seek(0)
    data = source.read()
    return lambda: io.BytesIO(data)


def _active_video_frames(
    video,
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> Iterator[tuple[int, Fraction, av.VideoFrame]]:
    """Yield active frames in presentation order with clip-relative PTS."""

    if source_factory is None:
        source_factory = _video_source_factory(video)
    source = source_factory()
    start_seconds, duration_seconds = video.get_active_trim_window()
    trim_start = _as_fraction(start_seconds)
    trim_duration = _as_fraction(duration_seconds)

    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The VIDEO input contains no video stream.")

        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("The video stream has no time base for PTS conversion.")

        stream_time_base = Fraction(stream.time_base)
        stream_start_pts = stream.start_time if stream.start_time is not None else 0
        stream_origin = Fraction(stream_start_pts) * stream_time_base
        active_start = stream_origin + trim_start
        active_end = active_start + trim_duration if trim_duration > 0 else None

        if trim_start > 0:
            seek_pts = stream_start_pts + int(trim_start / stream_time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        relative_origin = None
        previous_time = None
        frame_index = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(
                    "A decoded video frame has no PTS; exact timestamp sampling is unavailable."
                )

            frame_time_base = frame.time_base or stream.time_base
            if frame_time_base is None:
                raise ValueError(
                    "A decoded video frame has no time base for PTS conversion."
                )

            presentation_time = Fraction(frame.pts) * Fraction(frame_time_base)
            if presentation_time < active_start:
                continue
            if active_end is not None and presentation_time >= active_end:
                break
            if previous_time is not None and presentation_time < previous_time:
                raise ValueError("Video presentation timestamps are not monotonic.")

            if relative_origin is None:
                relative_origin = presentation_time
            relative_time = presentation_time - relative_origin
            previous_time = presentation_time
            yield frame_index, relative_time, frame
            frame_index += 1


def scan_video_frame_records(
    video,
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> list[VideoFrameRecord]:
    records = [
        VideoFrameRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            key_frame=bool(frame.key_frame),
        )
        for frame_index, timestamp, frame in _active_video_frames(
            video, source_factory
        )
    ]
    if not records:
        raise ValueError("The active VIDEO input contains no decodable frames.")
    return records


def _spacing_filter(
    records: Sequence[VideoFrameRecord], minimum_spacing: Fraction
) -> list[VideoFrameRecord]:
    selected: list[VideoFrameRecord] = []
    for record in sorted(records, key=lambda item: (item.timestamp, item.frame_index)):
        if not selected or record.timestamp - selected[-1].timestamp >= minimum_spacing:
            selected.append(record)
    return selected


def _evenly_thin(
    records: Sequence[VideoFrameRecord],
    count: int,
    *,
    single_from_end: bool,
) -> list[VideoFrameRecord]:
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return list(records)
    if count == 1:
        index = len(records) - 1 if single_from_end else len(records) // 2
        return [records[index]]

    denominator = count - 1
    last_index = len(records) - 1
    indices = [
        (position * last_index + denominator // 2) // denominator
        for position in range(count)
    ]
    return [records[index] for index in indices]


def _select_uniform_samples(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
    timestamp_format: str | None,
) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    zero_record = records[0]
    candidates = [record for record in records if record.frame_index != zero_record.frame_index]

    if maximum_frames == 0:
        combined = ([zero_record] if include_zero_time else []) + candidates
        selected = _spacing_filter(combined, minimum_spacing)
        return selected, [record.timestamp for record in selected]

    if not candidates:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    if include_zero_time and maximum_frames == 1:
        return [zero_record], [Fraction(0)]

    duration = records[-1].timestamp
    if duration <= 0:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    if include_zero_time:
        target_count = maximum_frames - 1
        targets = [
            duration * Fraction(position, target_count)
            for position in range(1, target_count + 1)
        ]
        selected_pairs = [(zero_record, Fraction(0))]
    else:
        target_count = maximum_frames
        targets = [
            duration * Fraction(position, target_count + 1)
            for position in range(1, target_count + 1)
        ]
        selected_pairs = []

    if timestamp_format is not None:
        targets = [
            round_video_timestamp(target, timestamp_format)
            for target in targets
        ]

    for target in targets:
        selected_pairs.append(
            (
                min(
                    candidates,
                    key=lambda record: (
                        abs(record.timestamp - target),
                        record.timestamp,
                        record.frame_index,
                    ),
                ),
                target,
            )
        )

    unique = {
        record.frame_index: (record, timestamp)
        for record, timestamp in selected_pairs
    }
    spaced = _spacing_filter(
        [record for record, _ in unique.values()],
        minimum_spacing,
    )
    timestamps_by_index = {
        record.frame_index: timestamp
        for record, timestamp in unique.values()
    }
    return spaced, [timestamps_by_index[record.frame_index] for record in spaced]


def _select_uniform_records(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
) -> list[VideoFrameRecord]:
    selected, _ = _select_uniform_samples(
        records,
        maximum_frames,
        include_zero_time,
        minimum_spacing,
        timestamp_format=None,
    )
    return selected


def _select_focused_samples(records: Sequence[VideoFrameRecord], maximum_frames: int, include_zero_time: bool, minimum_spacing: Fraction, timestamp_format: str | None, focus_areas: int, focus_one: float, focus_two: float, focus_three: float) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    if maximum_frames == 0:
        return _select_uniform_samples(records, maximum_frames, include_zero_time, minimum_spacing, timestamp_format)

    zero_record = records[0]
    candidates = [record for record in records if record.frame_index != zero_record.frame_index]
    if not candidates:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]
    if include_zero_time and maximum_frames == 1:
        return [zero_record], [Fraction(0)]

    duration = records[-1].timestamp
    if duration <= 0:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    targets = [
        _as_fraction(target) for target in focused_timeline_timestamps(
            maximum_frames, float(duration), focus_areas, focus_one, focus_two, focus_three, include_zero_time
        )
    ]
    if timestamp_format is not None:
        targets = [round_video_timestamp(target, timestamp_format) for target in targets]

    selected_pairs = []
    for target in targets:
        if include_zero_time and target == 0:
            selected_pairs.append((zero_record, Fraction(0)))
        else:
            selected_pairs.append((min(candidates, key=lambda record: (abs(record.timestamp - target), record.timestamp, record.frame_index)), target))

    unique = {record.frame_index: (record, timestamp) for record, timestamp in selected_pairs}
    spaced = _spacing_filter([record for record, _ in unique.values()], minimum_spacing)
    timestamps_by_index = {record.frame_index: timestamp for record, timestamp in unique.values()}
    return spaced, [timestamps_by_index[record.frame_index] for record in spaced]


def _select_keyframe_records(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
    keyframe_stride: int,
) -> list[VideoFrameRecord]:
    zero_record = records[0]
    raw_keyframes = [record for record in records if record.key_frame]
    candidates = raw_keyframes[::keyframe_stride]

    if include_zero_time:
        combined = [zero_record] + [
            record for record in candidates if record.frame_index != zero_record.frame_index
        ]
    else:
        combined = [record for record in candidates if record.timestamp > 0]

    spaced = _spacing_filter(combined, minimum_spacing)
    if maximum_frames == 0 or len(spaced) <= maximum_frames:
        return spaced

    if include_zero_time:
        zero = spaced[0]
        remaining = _evenly_thin(
            spaced[1:],
            maximum_frames - 1,
            single_from_end=True,
        )
        return [zero] + remaining

    return _evenly_thin(
        spaced,
        maximum_frames,
        single_from_end=False,
    )


def _select_video_frame_records_and_timestamps(
    records: Sequence[VideoFrameRecord],
    strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    timestamp_format: str | None,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    if strategy not in VIDEO_FRAME_SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported video-frame sampling strategy: {strategy}")
    if maximum_frames < 0:
        raise ValueError("maximum_frames must be zero or greater.")
    if minimum_spacing_seconds < 0:
        raise ValueError("minimum_spacing_seconds must be zero or greater.")
    if keyframe_stride < 1:
        raise ValueError("keyframe_stride must be at least one.")
    if not records:
        raise ValueError("No video-frame records are available for selection.")

    ordered = sorted(records, key=lambda item: (item.timestamp, item.frame_index))
    minimum_spacing = _as_fraction(minimum_spacing_seconds)

    if strategy == "uniform PTS":
        selected, output_timestamps = _select_uniform_samples(
            ordered,
            maximum_frames,
            include_zero_time,
            minimum_spacing,
            timestamp_format,
        )
    elif strategy == "focused PTS":
        selected, output_timestamps = _select_focused_samples(
            ordered, maximum_frames, include_zero_time, minimum_spacing, timestamp_format, focus_areas, focus_one, focus_two, focus_three
        )
    else:
        selected = _select_keyframe_records(
            ordered,
            maximum_frames,
            include_zero_time,
            minimum_spacing,
            keyframe_stride,
        )
        output_timestamps = [record.timestamp for record in selected]

    if not selected:
        raise ValueError("No video frames satisfy the selected sampling controls.")
    return selected, output_timestamps


def select_video_frame_records(
    records: Sequence[VideoFrameRecord],
    strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> list[VideoFrameRecord]:
    selected, _ = _select_video_frame_records_and_timestamps(
        records,
        strategy,
        maximum_frames,
        include_zero_time,
        minimum_spacing_seconds,
        keyframe_stride,
        timestamp_format=None,
        focus_areas=focus_areas,
        focus_one=focus_one,
        focus_two=focus_two,
        focus_three=focus_three,
    )
    return selected


def _frame_to_image(frame: av.VideoFrame) -> torch.Tensor:
    image = frame.to_ndarray(format="rgb24")
    rotation = getattr(frame, "rotation", 0) or 0
    if rotation:
        quarter_turns = int(round(rotation / 90.0))
        image = np.rot90(image, k=quarter_turns, axes=(0, 1)).copy()
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).to(dtype=torch.float32).div_(255.0)


def decode_selected_video_frames(
    video,
    records: Sequence[VideoFrameRecord],
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    expected = {record.frame_index: record for record in records}
    selected_images: dict[int, torch.Tensor] = {}
    last_index = max(expected)

    for frame_index, timestamp, frame in _active_video_frames(
        video, source_factory
    ):
        record = expected.get(frame_index)
        if record is not None:
            if timestamp != record.timestamp:
                raise ValueError("Video timestamps changed between selection and decoding.")
            selected_images[frame_index] = _frame_to_image(frame)
        if frame_index >= last_index:
            break

    missing = [record.frame_index for record in records if record.frame_index not in selected_images]
    if missing:
        raise ValueError("Selected video frames could not be decoded.")

    ordered_images = [selected_images[record.frame_index] for record in records]
    first_shape = ordered_images[0].shape
    if any(image.shape != first_shape for image in ordered_images[1:]):
        raise ValueError("Selected video frames have inconsistent image dimensions.")

    image_batch = torch.stack(ordered_images, dim=0)
    image_list = [image_batch[index : index + 1] for index in range(image_batch.shape[0])]
    return image_batch, image_list


def _rounded_units(timestamp: Fraction, units_per_second: int) -> int:
    if timestamp < 0:
        raise ValueError("Relative video timestamps cannot be negative.")
    numerator = timestamp.numerator * units_per_second
    denominator = timestamp.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def round_video_timestamp(
    timestamp: Fraction | float,
    timestamp_format: str,
) -> Fraction:
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError(f"Unsupported video timestamp format: {timestamp_format}")
    units_per_second = 10 if timestamp_format == "0.0s" else (
        100 if timestamp_format == "0.00s" else 1000
    )
    return Fraction(
        _rounded_units(_as_fraction(timestamp), units_per_second),
        units_per_second,
    )


def format_video_timestamp(timestamp: Fraction | float, timestamp_format: str) -> str:
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError(f"Unsupported video timestamp format: {timestamp_format}")

    value = _as_fraction(timestamp)
    if timestamp_format == "0.0s":
        total_deciseconds = _rounded_units(value, 10)
        seconds, deciseconds = divmod(total_deciseconds, 10)
        return f"{seconds}.{deciseconds}s"

    if timestamp_format == "0.00s":
        total_centiseconds = _rounded_units(value, 100)
        seconds, centiseconds = divmod(total_centiseconds, 100)
        return f"{seconds}.{centiseconds:02d}s"

    total_milliseconds = _rounded_units(value, 1000)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)

    if timestamp_format == "00.000s":
        return f"{total_seconds:02d}.{milliseconds:03d}s"

    total_minutes, seconds = divmod(total_seconds, 60)
    if timestamp_format == "MM:SS.mmm":
        return f"{total_minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if timestamp_format == "MM:SS:mmm":
        return f"{total_minutes:02d}:{seconds:02d}:{milliseconds:03d}"

    hours, minutes = divmod(total_minutes, 60)
    separator = "." if timestamp_format == "HH:MM:SS.mmm" else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def build_video_timeline_text(
    timestamps: Sequence[str],
    timeline_style: str,
    index_offset: int = 0,
) -> str:
    if timeline_style not in VIDEO_FRAME_TIMELINE_STYLES:
        raise ValueError(f"Unsupported video timeline style: {timeline_style}")
    if timeline_style == "timestamps only":
        lines = list(timestamps)
    elif timeline_style == "indexed":
        lines = [f"{index}: {timestamp}" for index, timestamp in enumerate(timestamps)]
    elif timeline_style == "H3 pictures":
        lines = [
            f"<Picture {index + index_offset}> at {timestamp}"
            for index, timestamp in enumerate(timestamps, start=1)
        ]
    else:
        lines = [
            (
                f"For the target video, at {timestamp} into the target video, "
                f"<Picture {index + index_offset}> (from [Shot {index}]) is fully referenced."
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
    return "\n".join(lines)


def build_structured_video_timeline_text(
    video_runtime: float,
    timestamps: Sequence[str],
    index_offset: int = 0,
) -> str:
    segment_count = len(timestamps)
    introduction = (
        f"Target video duration is {video_runtime:g} seconds divided into "
        f"{segment_count} segments."
    )
    references = [
        f"<Picture {index + index_offset}> at {timestamp}"
        for index, timestamp in enumerate(timestamps, start=1)
    ]
    if not references:
        return introduction
    if len(references) == 1:
        reference_text = references[0]
    else:
        reference_text = f"{', '.join(references[:-1])} and {references[-1]}"
    return f"{introduction} Reference each image with {reference_text}."


def _timeline_input_images(image_inputs) -> list[torch.Tensor]:
    """Flatten autogrow IMAGE inputs in numeric socket and batch order."""
    if not isinstance(image_inputs, dict):
        raise ValueError("Images to Video Timeline requires at least one connected image.")

    def socket_number(name):
        match = re.search(r"\d+", name)
        return int(match.group()) if match else 0

    images = []
    for name in sorted(image_inputs, key=socket_number):
        value = image_inputs[name]
        if value is None:
            continue
        if not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] < 1:
            raise ValueError("Images to Video Timeline inputs must be nonempty BHWC IMAGE batches.")
        images.extend(value[index:index + 1] for index in range(value.shape[0]))
    if not images:
        raise ValueError("Images to Video Timeline requires at least one connected image.")
    return images


def _timeline_image_outputs(image_inputs, resize_images: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Optionally normalize images for batching while preserving the ordered image list."""
    images = _timeline_input_images(image_inputs)
    if not resize_images:
        return torch.zeros((1, 64, 64, 3), dtype=images[0].dtype, device=images[0].device), images
    first_height, first_width = images[0].shape[1:3]
    max_channels = max(image.shape[-1] for image in images)
    normalized = []
    for image in images:
        if image.shape[-1] < max_channels:
            image = torch.nn.functional.pad(image, (0, max_channels - image.shape[-1]), value=1.0)
        if image.shape[1:3] != (first_height, first_width):
            method = "area" if first_height < image.shape[1] or first_width < image.shape[2] else "bicubic"
            image = resize_nchw(image.movedim(-1, 1), first_width, first_height, method).movedim(1, -1)
        normalized.append(image)
    image_batch = torch.cat(normalized, dim=0)
    return image_batch, [image_batch[index:index + 1] for index in range(image_batch.shape[0])]


def _truncated_normal_quantile(quantile: float, center: float) -> float:
    """Return a unit-interval Gaussian quantile centered at a local focus value."""
    deviation = 0.18
    root_two = math.sqrt(2.0)
    lower = 0.5 * (1.0 + math.erf(-center / (deviation * root_two)))
    upper = 0.5 * (1.0 + math.erf((1.0 - center) / (deviation * root_two)))
    target = lower + quantile * (upper - lower)
    low, high = 0.0, 1.0
    for _ in range(48):
        midpoint = (low + high) / 2.0
        value = 0.5 * (1.0 + math.erf((midpoint - center) / (deviation * root_two)))
        if value < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def focused_timeline_timestamps(count: int, duration: float, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, anchor_endpoints: bool = True) -> list[float]:
    """Place ordered timestamps across a duration with optional local focus peaks."""
    if count < 1:
        raise ValueError("Focused timeline requires at least one timestamp.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Images to Video Timeline duration must be finite and greater than zero.")
    if isinstance(focus_areas, bool) or focus_areas not in range(4):
        raise ValueError("Images to Video Timeline focus_areas must be an integer from 0 to 3.")
    focuses = (focus_one, focus_two, focus_three)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in focuses):
        raise ValueError("Images to Video Timeline focus values must be finite values from 0 to 1.")
    if anchor_endpoints and count == 1:
        return [0.0]
    if focus_areas == 0:
        denominator = count - 1 if anchor_endpoints else count + 1
        start = 0 if anchor_endpoints else 1
        return [duration * index / denominator for index in range(start, start + count)]

    movable_count = count - 2 if anchor_endpoints else count
    timestamps = [0.0] if anchor_endpoints else []
    for index in range(1, movable_count + 1):
        global_quantile = index / (movable_count + 1)
        section = min(int(global_quantile * focus_areas), focus_areas - 1)
        local_quantile = global_quantile * focus_areas - section
        local_position = _truncated_normal_quantile(local_quantile, focuses[section])
        timestamps.append(duration * (section + local_position) / focus_areas)
    if anchor_endpoints:
        timestamps.append(duration)
    return timestamps


def images_to_video_timeline(image_inputs, duration: float, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, resize_images: bool, timestamp_format: str, timeline_style: str, index_offset: int = 0) -> SampledVideoFrames:
    """Normalize supplied images and assign their manual video timeline timestamps."""
    image_batch, image_list = _timeline_image_outputs(image_inputs, resize_images)
    raw_timestamps = focused_timeline_timestamps(len(image_list), duration, focus_areas, focus_one, focus_two, focus_three)
    timestamps = [format_video_timestamp(timestamp, timestamp_format) for timestamp in raw_timestamps]
    return SampledVideoFrames(
        image_batch=image_batch,
        image_list=image_list,
        timestamps=timestamps,
        timestamps_text=", ".join(timestamps),
        timeline_text=build_video_timeline_text(timestamps, timeline_style, index_offset),
        video_runtime=duration,
        structured_timeline_text=build_structured_video_timeline_text(duration, timestamps, index_offset),
    )


def sample_video_frames_as_images(
    video,
    sampling_strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    timestamp_format: str,
    timeline_style: str,
    index_offset: int = 0,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> SampledVideoFrames:
    video_runtime = float(video.get_duration())
    source_factory = _video_source_factory(video)
    records = scan_video_frame_records(video, source_factory)
    selected, output_timestamps = _select_video_frame_records_and_timestamps(
        records,
        sampling_strategy,
        maximum_frames,
        include_zero_time,
        minimum_spacing_seconds,
        keyframe_stride,
        timestamp_format,
        focus_areas,
        focus_one,
        focus_two,
        focus_three,
    )
    image_batch, image_list = decode_selected_video_frames(
        video, selected, source_factory
    )
    timestamps = [
        format_video_timestamp(timestamp, timestamp_format)
        for timestamp in output_timestamps
    ]

    return SampledVideoFrames(
        image_batch=image_batch,
        image_list=image_list,
        timestamps=timestamps,
        timestamps_text=", ".join(timestamps),
        timeline_text=build_video_timeline_text(
            timestamps, timeline_style, index_offset
        ),
        video_runtime=video_runtime,
        structured_timeline_text=build_structured_video_timeline_text(
            video_runtime, timestamps, index_offset
        ),
    )
