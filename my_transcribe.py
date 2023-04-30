import argparse
import os
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

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
def transcription_with_backoff(**kwargs):
  try:
    transcriptions = []
    files = []
    count = 0

    audio_clips = glob.glob("output/*.mp3")
    if audio_clips <= 1:
      raise RuntimeError('No files found to transcribe.')

    for audio_clip_path in audio_clips:
      audio = AudioSegment.from_file(audio_clip_path, format='mp3')
      duration = len(audio) / 1000  # Convert milliseconds to seconds
      if duration < 0.15 or os.path.exists(audio_clip_path+'.'+format):
        continue

      count += 1
      with open(audio_clip_path, 'rb') as file:
        response = openai.Audio.transcribe(
          file=file,
          model="whisper-1",
          language="en",
          response_format='srt' #'text' is the default but not timestamped.
        )
      with open(audio_clip_path +'.'+ format,'w',  encoding="utf-8") as out:
        out.write(response)
  except Exception as e:
    raise e

def confirm_action(msg):
    while True:
        user_input = input(msg)
        if user_input.lower() == "yes":
            return True
        elif user_input.lower() == "no":
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='PodcastTranscriptionToSRT',
                    description='Sends the chunks of mp3 generated by my_parse.py' +
                                ' in output/*.mp3 to OpenAI whisper for' +
                                ' transcription to SRT format. Note that the SRT format' +
                                ' is frequently incorrect on timings and repetitions' +
                                ' from OpenAI.',
                    epilog='Run script_builder.py after completed')
    args = parser.parse_args()
    load_dotenv()
    
    openai.api_key = os.getenv('OPENAI_API_TOKEN')
    confirm_action('This action will load a large number of files to OpenAI\'s whisper endpoint'
                   + ' automatically. This may be slow and expensive. Continue?')
    transcription_with_backoff()
    print(transcription_with_backoff.retry.statistics)

    