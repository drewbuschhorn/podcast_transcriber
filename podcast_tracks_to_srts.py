import argparse
import os
from typing import Dict
from dotenv import load_dotenv
import openai
import datetime
import glob
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from pydub import AudioSegment
import logging

from helper import confirm_action, open_file_wdirs, sanitize_file_name

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
def transcription_with_backoff(**kwargs : Dict[str, str]):
  output_path = kwargs['output_dir'] + '/' + kwargs['filename']
  try:
    transcriptions = []
    files = []
    count = 0
    format = 'srt' #'text' is the default but not timestamped.

    audio_clips = glob.glob(output_path + "/*.mp3")
    if len(audio_clips) <= 1:
      raise RuntimeError('No files found to transcribe.')

    for audio_clip_path in audio_clips:
      audio = AudioSegment.from_file(audio_clip_path, format='mp3')
      duration = len(audio) / 1000  # Convert milliseconds to seconds
      if duration < 0.15 or os.path.exists(audio_clip_path+'.'+format):
        continue

      count += 1
      with open_file_wdirs(audio_clip_path, 'rb') as file:
        response = openai.Audio.transcribe(
          file=file,
          model="whisper-1",
          language="en",
          response_format=format
        )
      with open_file_wdirs(audio_clip_path +'.'+ format,'w',  encoding="utf-8") as out:
        out.write(response)
  except Exception as e:
    logging.debug(e)
    raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='podcast_tracks_to_srts',
                    description='Sends the chunks of mp3 generated by my_parse.py' +
                                ' in output/[input_file_name]/*.mp3 to OpenAI whisper for' +
                                ' transcription to SRT format. Note that the SRT format' +
                                ' is frequently incorrect on timings and repetitions' +
                                ' from OpenAI.',
                    epilog='Run podcast_srts_to_transcript.py after completed')
    parser.add_argument('--file', metavar='-f', required=True,
                    help='file name of podcast file (to find working direction in output directoy)')
    parser.add_argument('--output', metavar='-o', default='output', required = False,
                    help='output directory to work in')
    parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Set the logging level')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    load_dotenv()
    
    openai.api_key = os.getenv('OPENAI_API_TOKEN')
    confirm_action('This action will load a large number of files to OpenAI\'s whisper endpoint'
                   + ' automatically. This may be slow and expensive. Continue? [Y/N]: ')
    transcription_with_backoff(filename = sanitize_file_name(args.file), output_dir = args.output)
    logging.info(transcription_with_backoff.retry.statistics)

    