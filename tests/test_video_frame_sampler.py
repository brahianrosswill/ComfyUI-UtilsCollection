import io as stdlib_io
import pathlib
import sys
import types
from fractions import Fraction

import numpy as np
import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_frame_sampler_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_frame_sampler_test import image_helpers
from utils_collection_video_frame_sampler_test.image_helpers import (
    VideoFrameRecord,
    build_video_timeline_text,
    format_video_timestamp,
    parse_video_timestamp,
    parse_video_timestamps,
    sample_video_frames_as_images,
    scan_video_frame_records,
    select_video_frame_records,
)


@pytest.mark.parametrize("value, expected", [
    ("01:02:03.250", Fraction(14893, 4)),
    ("02:03:250", Fraction(493, 4)),
    ("02:03.250", Fraction(493, 4)),
    ("00.125s", Fraction(1, 8)),
    (1.25, Fraction(5, 4)),
])
def test_parse_video_timestamp_formats(value, expected):
    assert parse_video_timestamp(value) == expected


def test_parse_video_timestamps_flattens_delimited_and_nested_values():
    assert parse_video_timestamps([["0; 1.2"], ("2.5s",)]) == [Fraction(0), Fraction(6, 5), Fraction(5, 2)]


@pytest.mark.parametrize("value", [True, -1, float("inf"), "", "1;;2", "00:00:60.0"])
def test_parse_video_timestamps_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_video_timestamps(value)


def test_parse_video_timestamps_rejects_decreasing_order():
    with pytest.raises(ValueError, match="earlier"):
        parse_video_timestamps([1, 0])
from utils_collection_video_frame_sampler_test.image_nodes import (
    UC_SampleVideoFramesAsImages,
)


def _records(times, keyframes=None):
    if keyframes is None:
        keyframes = [True] * len(times)
    return [
        VideoFrameRecord(index, Fraction(str(timestamp)), key_frame)
        for index, (timestamp, key_frame) in enumerate(zip(times, keyframes))
    ]


def test_schema_exposes_batch_list_and_aligned_metadata_outputs():
    schema = UC_SampleVideoFramesAsImages.define_schema()

    assert schema.node_id == "UC_SampleVideoFramesAsImages"
    assert schema.display_name == "Sample Video Frames (Images)"
    assert [value.id for value in schema.inputs] == [
        "video",
        "sampling_strategy",
        "maximum_frames",
        "include_zero_time",
        "minimum_spacing_seconds",
        "keyframe_stride",
        "timestamp_format",
        "timeline_style",
    ]
    assert schema.inputs[1].default == "codec keyframes"
    assert schema.inputs[2].default == 16
    assert schema.inputs[3].default is True
    assert schema.inputs[4].default == 0.25
    assert schema.inputs[5].default == 1
    assert schema.inputs[6].default == "00.000s"
    assert "0.00s" in schema.inputs[6].options
    assert "00.00s" not in schema.inputs[6].options
    assert schema.inputs[7].default == "H3 alignment prefix"
    assert [output.id for output in schema.outputs] == [
        "image_batch",
        "images",
        "timestamps",
        "timestamps_text",
        "timeline_text",
    ]
    assert [output.is_output_list for output in schema.outputs] == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_zero_time_is_output_position_zero_and_spacing_removes_near_duplicate():
    selected = select_video_frame_records(
        _records([0, 0.01, 0.25, 0.49, 0.5]),
        "codec keyframes",
        maximum_frames=0,
        include_zero_time=True,
        minimum_spacing_seconds=0.25,
        keyframe_stride=1,
    )

    assert [record.frame_index for record in selected] == [0, 2, 4]
    assert [record.timestamp for record in selected] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
    ]


def test_keyframe_stride_precedes_even_full_timeline_count_limiting():
    selected = select_video_frame_records(
        _records(range(10)),
        "codec keyframes",
        maximum_frames=3,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=2,
    )

    assert [record.frame_index for record in selected] == [0, 2, 8]


def test_uniform_pts_uses_irregular_presentation_times():
    records = _records([0, 0.1, 0.9, 2.1, 4.0])

    with_zero = select_video_frame_records(
        records,
        "uniform PTS",
        maximum_frames=3,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
    )
    without_zero = select_video_frame_records(
        records,
        "uniform PTS",
        maximum_frames=2,
        include_zero_time=False,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
    )

    assert [record.frame_index for record in with_zero] == [0, 3, 4]
    assert [record.frame_index for record in without_zero] == [2, 3]


@pytest.mark.parametrize(
    ("timestamp_format", "expected"),
    [
        ("HH:MM:SS.mmm", "00:02:03.456"),
        ("HH:MM:SS:mmm", "00:02:03:456"),
        ("MM:SS.mmm", "02:03.456"),
        ("MM:SS:mmm", "02:03:456"),
        ("00.000s", "123.456s"),
        ("0.00s", "123.46s"),
    ],
)
def test_timestamp_formats_are_deterministic(timestamp_format, expected):
    assert format_video_timestamp(Fraction(123456, 1000), timestamp_format) == expected


def test_two_decimal_seconds_use_minimal_integer_width():
    assert format_video_timestamp(Fraction(0), "0.00s") == "0.00s"
    assert format_video_timestamp(Fraction(1234, 1000), "0.00s") == "1.23s"
    assert format_video_timestamp(Fraction(12304, 1000), "0.00s") == "12.30s"
    assert format_video_timestamp(Fraction(1234, 1000), "00.000s") == "01.234s"


def test_timestamp_rounding_carries_into_the_next_minute():
    timestamp = Fraction(599996, 10000)

    assert format_video_timestamp(timestamp, "MM:SS.mmm") == "01:00.000"
    assert format_video_timestamp(timestamp, "MM:SS:mmm") == "01:00:000"


def test_timeline_reuses_formatted_timestamp_literals_without_unit_rewriting():
    timestamps = ["00.000s", "2.50s"]

    assert build_video_timeline_text(timestamps, "timestamps only") == (
        "00.000s\n2.50s"
    )
    assert build_video_timeline_text(timestamps, "indexed") == (
        "0: 00.000s\n1: 2.50s"
    )
    assert build_video_timeline_text(timestamps, "H3 pictures") == (
        "<Picture 1> at 00.000s\n<Picture 2> at 2.50s"
    )
    assert build_video_timeline_text(timestamps, "H3 alignment prefix") == (
        "For the target video, at 00.000s into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.\n"
        "For the target video, at 2.50s into the target video, "
        "<Picture 2> (from [Shot 2]) is fully referenced."
    )


class _FakeFrame:
    def __init__(self, pts, key_frame, value, time_base=Fraction(1, 1000)):
        self.pts = pts
        self.time_base = time_base
        self.key_frame = key_frame
        self.rotation = 0
        self._value = value

    def to_ndarray(self, format):
        assert format == "rgb24"
        return np.full((2, 3, 3), self._value, dtype=np.uint8)


class _FakeStream:
    def __init__(self, start_time=0, time_base=Fraction(1, 1000)):
        self.start_time = start_time
        self.time_base = time_base


class _FakeContainer:
    def __init__(self, frames, stream):
        self._frames = frames
        self.streams = types.SimpleNamespace(video=[stream])
        self.seek_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def seek(self, pts, stream, backward, any_frame):
        self.seek_calls.append((pts, stream, backward, any_frame))

    def decode(self, stream):
        assert stream is self.streams.video[0]
        return iter(self._frames)


class _FakeVideo:
    def __init__(self, start_time=0.0, duration=0.0):
        self._trim = (start_time, duration)
        self.source_calls = 0

    def get_stream_source(self):
        self.source_calls += 1
        return stdlib_io.BytesIO()

    def get_active_trim_window(self):
        return self._trim


def test_pts_scan_normalizes_the_first_visible_trim_frame(monkeypatch):
    frames = [
        _FakeFrame(1400, True, 1),
        _FakeFrame(1500, False, 2),
        _FakeFrame(1750, True, 3),
        _FakeFrame(2499, True, 4),
        _FakeFrame(2500, True, 5),
    ]
    stream = _FakeStream(start_time=1000)
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    records = scan_video_frame_records(_FakeVideo(start_time=0.5, duration=1.0))

    assert [record.frame_index for record in records] == [0, 1, 2]
    assert [record.timestamp for record in records] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(999, 1000),
    ]
    assert [record.key_frame for record in records] == [False, True, True]


def test_sampler_outputs_aligned_batch_list_and_metadata(monkeypatch):
    frames = [
        _FakeFrame(0, True, 0),
        _FakeFrame(10, True, 10),
        _FakeFrame(250, True, 20),
        _FakeFrame(1000, True, 30),
    ]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    video = _FakeVideo()
    sampled = sample_video_frames_as_images(
        video,
        "codec keyframes",
        maximum_frames=0,
        include_zero_time=True,
        minimum_spacing_seconds=0.25,
        keyframe_stride=1,
        timestamp_format="00.000s",
        timeline_style="H3 pictures",
    )

    assert sampled.image_batch.shape == (3, 2, 3, 3)
    assert len(sampled.image_list) == 3
    assert all(image.shape == (1, 2, 3, 3) for image in sampled.image_list)
    assert sampled.image_list[1].data_ptr() == sampled.image_batch[1:2].data_ptr()
    assert sampled.timestamps == ["00.000s", "00.250s", "01.000s"]
    assert sampled.timestamps_text == "00.000s, 00.250s, 01.000s"
    assert sampled.timeline_text == (
        "<Picture 1> at 00.000s\n"
        "<Picture 2> at 00.250s\n"
        "<Picture 3> at 01.000s"
    )
    assert torch.allclose(sampled.image_batch[1], torch.full((2, 3, 3), 20 / 255))
    assert video.source_calls == 1


def test_missing_pts_is_rejected_instead_of_estimated(monkeypatch):
    frames = [_FakeFrame(None, True, 0)]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    with pytest.raises(ValueError, match="has no PTS"):
        scan_video_frame_records(_FakeVideo())
