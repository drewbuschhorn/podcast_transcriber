import argparse
import copy
import logging
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

from helper import confirm_action, open_file_wdirs, sanitize_file_name

FIXUP_PROMPT = """
Identify typos to correct in the following transcript of a podcast about the video game Dwarf Fortress. Make the corrections. Make the text more readable and natural. Do not change any content between '[' and ']'. Do not remove lines starting with '[interjection]'.
----

{i}
"""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
def fixup_with_backoff(**kwargs : Dict[str, str]):
  output_path = kwargs['output_path']
  try:
    chunk_size = 2000
    chunks_of_less_than_size_x = []
    
    line_chunks = []
    with open_file_wdirs(output_path +'/real.txt', 'r') as file:
      line_chunks = file.readlines()
    
    current_chunk = ""
    for i in line_chunks:
       if len(current_chunk)+len(i)+1 < chunk_size:
          current_chunk += i
       else:
          chunks_of_less_than_size_x.append(current_chunk)
          current_chunk = i
    if chunks_of_less_than_size_x[len(chunks_of_less_than_size_x)-1] != current_chunk:
       chunks_of_less_than_size_x.append(current_chunk)
      
    with open_file_wdirs(output_path +'/fixup.txt','w') as out:
      for i in chunks_of_less_than_size_x:
        logging.info("sending to openai: \n" + str(i))
        use_prompt = copy.deepcopy(FIXUP_PROMPT)

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
                {"role": "user", 
                 "content": use_prompt.format(i=i)},
            ]
        )

        if response['choices'][0]["finish_reason"] != "stop":
          raise RuntimeError('OpenAI error: ' + str([response['choices'][0]["finish_reason"], i]))
        out.write(response['choices'][0]['message']['content'] + "\n")
        out.flush()
       
  except Exception as e:
    logging.info (e)
    raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='podcast_transcript_gpt_fixup',
                    description='Sends the chunks of text generated by script_builder.py' +
                                ' in output/[input_file_name]/real.txt to OpenAI ChatGPT for' +
                                ' transcription fixup. '
                    )
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
    confirm_action('This action will load a large number of files to OpenAI\'s ChatGPT endpoint'
                   + ' automatically. This may be slow and expensive. Continue? [Y/N]: ')
    fixup_with_backoff(output_path = args.output + '\\' + sanitize_file_name(args.file))
    logging.info(fixup_with_backoff.retry.statistics)

  