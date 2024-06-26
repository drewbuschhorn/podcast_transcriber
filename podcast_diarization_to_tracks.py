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
from typing import Any, Text, Mapping, Optional
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pydub import AudioSegment
import torch
from tqdm import tqdm

from helper import crop, get_embeddings, open_file_wdirs, sanitize_file_name

# Monkey patch class Audio for fix to https://github.com/pyannote/pyannote-audio/pull/1326/commits/8c238a5b199420c3ce3911a2d39ced3b9ca593e4
# Terrible patching from helpers.py
Audio.crop = crop
SpeakerDiarization.get_embeddings = get_embeddings
# end patch

class YourCustomHook:
    def __init__(self):
        self.progress_bar = tqdm(desc="Processing", ncols=100, unit="step")

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        # If 'total' is provided in one of the hook calls, update the progress bar's total
        if total is not None and self.progress_bar.total is None:
            self.progress_bar.total = total
            self.progress_bar.refresh()

        # Update the progress bar with the number of completed steps
        if completed is not None:
            self.progress_bar.update(completed - self.progress_bar.n)

        # Optionally, you can also set post-fix information to display the current step name
        self.progress_bar.set_postfix_str(f"Current step: {step_name}", refresh=True)
        
        if total == completed:
            self.progress_bar.total = None
            print (f"\n{step_name} completed\n")

    def close(self):
        # Close the progress bar once done
        self.progress_bar.close()

def do_diarization(filename, hf_token, output_dir, num_speakers = None):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=hf_token)
    file_sub_directory_name = sanitize_file_name(filename)
    output_path = output_dir + '/' + file_sub_directory_name
    os.makedirs(output_path, exist_ok=True)

    for i in range(torch.cuda.device_count()):
        print(f"{i}: {torch.cuda.get_device_name(i)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. apply pretrained pipeline
    pipeline = pipeline.to(device) # Set to use first found CUDA GPU
    with open_file_wdirs(filename, 'rb') as file:
        logging.info(f"Sending {filename} to pipeline")
        diarization = pipeline(file, num_speakers=num_speakers, hook=YourCustomHook())
        logging.info(f"Sending {file} from file")
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