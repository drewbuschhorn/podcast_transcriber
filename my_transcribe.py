import os
import openai
import datetime
import glob
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from pydub import AudioSegment

print(datetime.datetime.now())

openai.api_key = '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'

audio_clips = [
    "output/2745334ms-2773313ms_speaker_SPEAKER_01.mp3",
    "output/207689ms-236680ms_speaker_SPEAKER_00.mp3",
    "output/2541990ms-2571387ms_speaker_SPEAKER_00.mp3",
        "output/577757ms-585452ms_speaker_SPEAKER_00.mp3",
            "output/664140ms-669878ms_speaker_SPEAKER_01.mp3",
                "output/1833139ms-1838843ms_speaker_SPEAKER_00.mp3",
]

format = 'srt' #'text' is the default but not timestamped.

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
def transcription_with_backoff(**kwargs):
  audio_clips = glob.glob("output/*.mp3")

  transcriptions = []
  files = []

  count = 0

  try:
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
          response_format=format
        )
        print ([audio_clip_path, response])
      with open(audio_clip_path+'.'+format,'w',  encoding="utf-8") as out:
        print(response)
        out.write(response)
  except Exception as e:
    print ("Error:")
    print (e)
    print ("Count:")
    print (count)
    raise e

  print (count)
  print (datetime.datetime.now())


transcription_with_backoff()
print(transcription_with_backoff.retry.statistics)