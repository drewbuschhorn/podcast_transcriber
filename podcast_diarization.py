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
from pyannote.audio import Pipeline
from pydub import AudioSegment

from helper import sanitize_file_name

def do_diarization(filename, hf_token, output_dir, num_speakers = None):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    file_sub_directory_name = sanitize_file_name(filename)
    output_path = output_dir + '/' + file_sub_directory_name

    # 4. apply pretrained pipeline
    pipeline = pipeline.to(0) # Set to use first found CUDA GPU
    with open(filename, 'rb') as file:
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


    with open(output_path + "/durations_and_speakers.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Duration (s)", "Speaker", "start_ms", "end_ms"])
        for duration, speaker, start_ms, end_ms in audio_snippets:
            writer.writerow([duration, speaker, start_ms, end_ms])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='podcast_diarization',
                    description='Parses an MP3 file into separate speaker chunks'+
                                ' by diarization. Generates a lot of of files in output/[input_file_name]/'+
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