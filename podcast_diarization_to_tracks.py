# Taken from: https://github.com/pyannote/pyannote-audio https://huggingface.co/pyannote/speaker-diarization
# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline

# IMPORTANT! APPLY https://github.com/pyannote/pyannote-audio/pull/1326/files
# if not already merged

import argparse
import csv
import logging
import os, re
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Audio
from pyannote.audio.core.io import get_torchaudio_info, AudioFile, AudioFileDocString
from pydub import AudioSegment

from helper import open_file_wdirs, sanitize_file_name

# Monkey patch class Audio for fix to https://github.com/pyannote/pyannote-audio/pull/1326/commits/8c238a5b199420c3ce3911a2d39ced3b9ca593e4
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
Audio.crop = crop
# end patch

def do_diarization(filename, hf_token, output_dir, num_speakers = None):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    file_sub_directory_name = sanitize_file_name(filename)
    output_path = output_dir + '/' + file_sub_directory_name
    os.makedirs(output_path, exist_ok=True)

    # 4. apply pretrained pipeline
    pipeline = pipeline.to(0) # Set to use first found CUDA GPU
    with open_file_wdirs(filename, 'rb') as file:
        diarization = pipeline(file, num_speakers=num_speakers)
        audio_file = AudioSegment.from_file(file)


    # 5. print the result
    audio_snippets = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        duration = end_ms - start_ms
        filename = output_path + f"/{start_ms}ms-{end_ms}ms_speaker_{speaker}.mp3"
        audio_snippet = audio_file[start_ms:end_ms]
        audio_snippet.export(filename, format="mp3")
        audio_snippets.append((duration, speaker, start_ms, end_ms))


    with open_file_wdirs(output_path + "/durations_and_speakers.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Duration (s)", "Speaker", "start_ms", "end_ms"])
        for duration, speaker, start_ms, end_ms in audio_snippets:
            writer.writerow([duration, speaker, start_ms, end_ms])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='podcast_diarization_to_tracks',
                    description='Parses an MP3 file into separate speaker chunks'+
                                ' by diarization. Generates a lot of files in output/[input_file_name]/'+
                                ' including a file durations_and_speakers.csv',
                    epilog='Run podcast_tracks_to_srts.py after completed')
    parser.add_argument('--file', metavar='-f', required=True,
                    help='input file to process')
    parser.add_argument('--num_speakers', metavar='-s', type=int, default=None, required = False,
                    help='best guess on number of speakers in file, or don\'t specify')
    parser.add_argument('--output', metavar='-o', default='output', required = False,
                    help='output directory to work in')
    parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Set the logging level')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    load_dotenv()

    do_diarization(args.file, os.getenv('HUGGING_FACE_TOKEN'), args.output, args.num_speakers)