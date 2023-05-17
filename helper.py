import os
import re

import functools
import itertools
import math
from typing import Callable, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.signal import binarize
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.io import get_torchaudio_info, AudioFile, AudioFileDocString
import math
import random
import warnings
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Text, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor


def sanitize_file_name(file_name):
    file_name = os.path.basename(file_name)
    # Remove non-alphanumeric characters (except dot) using regular expressions
    sanitized_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)
    # Remove leading and trailing underscores
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name

def confirm_action(msg, skip=False):
    if skip:
        return

    while True:
        user_input = input(msg)
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def open_file_wdirs(path, mode="wb", newline=None, encoding=None):
    # Ensure the parent directory exists
    dirname = os.path.dirname(path)
    if (dirname):
        os.makedirs(dirname, exist_ok=True)

    # Open file for binary writing
    file = open(path, mode, newline=newline, encoding=encoding)

    return file

### Monkey patching

def crop(
    self,
    file: AudioFile,
    segment: Segment,
    duration: Optional[float] = None,
    mode="raise",
) -> Tuple[Tensor, int]:
    """Fast version of self(file).crop(segment, **kwargs)

    Parameters
    ----------
    file : AudioFile
        Audio file.
    segment : `pyannote.core.Segment`
        Temporal segment to load.
    duration : float, optional
        Overrides `Segment` 'focus' duration and ensures that the number of
        returned frames is fixed (which might otherwise not be the case
        because of rounding errors).
    mode : {'raise', 'pad'}, optional
        Specifies how out-of-bounds segments will behave.
        * 'raise' -- raise an error (default)
        * 'pad' -- zero pad

    Returns
    -------
    waveform : (channel, time) torch.Tensor
        Waveform
    sample_rate : int
        Sample rate

    """
    file = self.validate_file(file)

    if "waveform" in file:
        waveform = file["waveform"]
        frames = waveform.shape[1]
        sample_rate = file["sample_rate"]

    elif "torchaudio.info" in file:
        info = file["torchaudio.info"]
        frames = info.num_frames
        sample_rate = info.sample_rate

    else:
        info = get_torchaudio_info(file)
        frames = info.num_frames
        sample_rate = info.sample_rate

    channel = file.get("channel", None)

    # infer which samples to load from sample rate and requested chunk
    start_frame = math.floor(segment.start * sample_rate)

    if duration:
        num_frames = math.floor(duration * sample_rate)
        end_frame = start_frame + num_frames

    else:
        end_frame = math.floor(segment.end * sample_rate)
        num_frames = end_frame - start_frame

    if mode == "raise":

        if num_frames > frames:
            raise ValueError(
                f"requested fixed duration ({duration:6f}s, or {num_frames:d} frames) is longer "
                f"than file duration ({frames / sample_rate:.6f}s, or {frames:d} frames)."
            )

        if end_frame > frames + math.ceil(self.PRECISION * sample_rate):
            raise ValueError(
                f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                f"lies outside of {file.get('uri', 'in-memory')} file bounds [0., {frames / sample_rate:.6f}s] ({frames:d} frames)."
            )
        else:
            end_frame = min(end_frame, frames)
            start_frame = end_frame - num_frames

        if start_frame < 0:
            raise ValueError(
                f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                f"lies outside of {file.get('uri', 'in-memory')} file bounds [0, {frames / sample_rate:.6f}s] ({frames:d} frames)."
            )

    elif mode == "pad":
        pad_start = -min(0, start_frame)
        pad_end = max(end_frame, frames) - frames
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, frames)
        num_frames = end_frame - start_frame

    if "waveform" in file:
        data = file["waveform"][:, start_frame:end_frame]

    else:
        try:
            data, _ = torchaudio.load(
                file["audio"], frame_offset=start_frame, num_frames=num_frames
            )
            # rewind if needed
            if isinstance(file["audio"], IOBase):
                file["audio"].seek(0)
        except RuntimeError:

            if isinstance(file["audio"], IOBase):
                msg = "torchaudio failed to seek-and-read in file-like object."
                raise RuntimeError(msg)

            msg = (
                f"torchaudio failed to seek-and-read in {file['audio']}: "
                f"loading the whole file instead."
            )

            warnings.warn(msg)
            waveform, sample_rate = self.__call__(file)
            data = waveform[:, start_frame:end_frame]

            # storing waveform and sample_rate for next time
            # as it is very likely that seek-and-read will
            # fail again for this particular file
            file["waveform"] = waveform
            file["sample_rate"] = sample_rate

    if channel is not None:
        data = data[channel : channel + 1, :]

    # pad with zeros
    if mode == "pad":
        # fix for #1324
        # F.pad here is padding the last dimension with pad_start at the beginning and pad_end at the end
        # data.shape[-1] is the len of the last dimension
        # choosing to pad towards the end to be consistent with how torchaudio.load() works 
        pad_end = num_frames - data.shape[-1] - pad_start
        data = F.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)

##

def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def get_embeddings(
    self,
    file,
    binary_segmentations: SlidingWindowFeature,
    exclude_overlap: bool = False,
    hook: Optional[Callable] = None,
):
    """Extract embeddings for each (chunk, speaker) pair

    Parameters
    ----------
    file : AudioFile
    binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        Binarized segmentation.
    exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        In case non-overlapping speech is too short, use the whole speech.
    hook: Optional[Callable]
        Called during embeddings after every batch to report the progress

    Returns
    -------
    embeddings : (num_chunks, num_speakers, dimension) array
    """

    # when optimizing the hyper-parameters of this pipeline with frozen
    # "segmentation.threshold", one can reuse the embeddings from the first trial,
    # bringing a massive speed up to the optimization process (and hence allowing to use
    # a larger search space).
    if self.training:
        # we only re-use embeddings if they were extracted based on the same value of the
        # "segmentation.threshold" hyperparameter or if the segmentation model relies on
        # `powerset` mode
        cache = file.get("training_cache/embeddings", dict())
        if ("embeddings" in cache) and (
            self._segmentation.model.specifications.powerset
            or (cache["segmentation.threshold"] == self.segmentation.threshold)
        ):
            return cache["embeddings"]

    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    if exclude_overlap:
        # minimum number of samples needed to extract an embedding
        # (a lower number of samples would result in an error)
        min_num_samples = self._embedding.min_num_samples

        # corresponding minimum number of frames
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

        # zero-out frames with overlapping speech
        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames,
            binary_segmentations.sliding_window,
        )

    else:
        min_num_frames = -1
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data, binary_segmentations.sliding_window
        )

    def iter_waveform_and_mask():
        for (chunk, masks), (_, clean_masks) in zip(
            binary_segmentations, clean_segmentations
        ):
            # chunk: Segment(t, t + duration)
            # masks: (num_frames, local_num_speakers) np.ndarray

            waveform, _ = self._audio.crop(
                file,
                chunk,
                duration=duration,
                mode="pad",
            )
            # waveform: (1, num_samples) torch.Tensor

            # mask may contain NaN (in case of partial stitching)
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

            for mask, clean_mask in zip(masks.T, clean_masks.T):
                # mask: (num_frames, ) np.ndarray

                if np.sum(clean_mask) > min_num_frames:
                    used_mask = clean_mask
                else:
                    used_mask = mask

                yield waveform[None], torch.from_numpy(used_mask)[None]
                # w: (1, 1, num_samples) torch.Tensor
                # m: (1, num_frames) torch.Tensor

    batches = batchify(
        iter_waveform_and_mask(),
        batch_size=self.embedding_batch_size,
        fillvalue=(None, None),
    )

    batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)

    embedding_batches = []

    for i, batch in enumerate(batches, 1):
        waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

        # Drew Hack for mismatched tensor errors
        max_len = max([x.squeeze().numel() for x in waveforms])
        waveforms = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in waveforms]
        # Drew Hack end

        waveform_batch = torch.vstack(waveforms)
        # (batch_size, 1, num_samples) torch.Tensor

        mask_batch = torch.vstack(masks)
        # (batch_size, num_frames) torch.Tensor

        embedding_batch: np.ndarray = self._embedding(
            waveform_batch, masks=mask_batch
        )
        # (batch_size, dimension) np.ndarray

        embedding_batches.append(embedding_batch)

        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=i)

    embedding_batches = np.vstack(embedding_batches)

    embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

    # caching embeddings for subsequent trials
    # (see comments at the top of this method for more details)
    if self.training:
        if self._segmentation.model.specifications.powerset:
            file["training_cache/embeddings"] = {
                "embeddings": embeddings,
            }
        else:
            file["training_cache/embeddings"] = {
                "segmentation.threshold": self.segmentation.threshold,
                "embeddings": embeddings,
            }

    return embeddings