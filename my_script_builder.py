import collections
from glob import glob
import os
from parse import parse
import pysrt
from thefuzz import fuzz

def convert_ms_to_hms(milliseconds):
    seconds = int((milliseconds / 1000) % 60)
    minutes = int((milliseconds / (1000 * 60)) % 60)
    hours = int((milliseconds / (1000 * 60 * 60)) % 24)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def write_line(file, queue, line):
  if is_a_repeat(queue, line):
    print (['repeat', line, queue])
    return

  queue.append(line)
  file.write(line + " ")

def is_a_repeat(queue, line):
  for index in range(len(queue)):
    queue_item = queue[index]
    fuzz_value = fuzz.partial_ratio(line, queue_item)
    step_threshold = (0.99**(index+1))*100
    if  fuzz_value > step_threshold:
      print (['fuzz', line, queue_item, index, step_threshold,  fuzz_value])
      return True
  return False


files = [os.path.normpath(i) for i in glob('output/*.mp3.srt')]

results = []
for filename in files:
  result = parse('output\\{start_ms:d}ms-{end_ms:d}ms_speaker_{speaker_id:S}.mp3.srt', filename)
  if result['speaker_id'] == "SPEAKER_00":
    speaker = "DREW"
  else:
    speaker = "KRISTIN"

  results.append({
    "parses": {
      'start_ms':  result['start_ms'],
      'end_ms': result['end_ms'],
      'speaker_id': speaker
    },
    "filename": filename
  })

results.sort(key=lambda x: x['parses']['start_ms'])

# Read an SRT file
with open('output/debug.txt', 'w') as out:
  for file in results:
    subs = pysrt.open(file['filename'])
    out.write(file['parses']['speaker_id'] + ":" 
              + str(file['parses']['start_ms']) 
              + "-" 
              + str(file['parses']['end_ms']) 
              + "\n")
    for sub in subs:
      out.write(
        str(file['parses']['start_ms'] + sub.start.ordinal) 
        + ":" 
        + str(file['parses']['start_ms'] + sub.end.ordinal)
        + "-"
        + str(sub.start.ordinal) 
        + ":" 
        + str(sub.end.ordinal) 
        + " " + sub.text + "\n" + str(sub) + "\n")
    out.write('\n')


# Read an SRT file
with open('output/real.txt', 'w') as out:
  line_record = collections.deque([], maxlen=10)
  result_iterator = iter(results)
  for result in result_iterator:
    file = result
    info = file['parses']

    try:
      interjection_files = list(filter(lambda x: True if x['parses']['start_ms'] > info['start_ms']+1 
                              and x['parses']['end_ms'] < info['end_ms'] else False, 
                              results))
      interjection_lines = []
      for j in interjection_files:
        results.remove(j)

        interjection_line = '\n[interjection] '
        # out.write(str(convert_ms_to_hms(j['parses']['start_ms'])) + " - ")
        interjection_line += j['parses']['speaker_id'] + ": "
        _subs = pysrt.open(j['filename'])
        for k in _subs:
          if not is_a_repeat(line_record, k.text):
            interjection_line += k.text
            interjection_lines.append({
              'start_time': j['parses']['start_ms'] + k.start.ordinal,
              'end_time': j['parses']['end_ms'],
              'origin': j,
              'raw_line': k.text,
              'line': interjection_line
            })
    except IndexError:
      pass
    except Exception as e:
      raise e
  
    print ([['interjections'],interjection_lines])

    subs = pysrt.open(file['filename'])
    show_original_speaker_after_interjection = False
    first_line = True
    for line in subs:
      if len(line.text.strip()) > 1:
        if first_line and not is_a_repeat(line_record, line.text.strip()):
          out.write("\n["+str(convert_ms_to_hms(info['start_ms'])) + "] ")
          out.write(info['speaker_id'] + ":\n")
          first_line = False

        if show_original_speaker_after_interjection:
          out.write("\n[interjection] " + info['speaker_id'] + ": ")

        write_line(out, line_record, line.text.strip())
        show_original_speaker_after_interjection = False
        for l in interjection_lines:
          if l['start_time'] >= file['parses']['start_ms'] and l['end_time'] <= file['parses']['end_ms']:
            if l['start_time'] >= file['parses']['start_ms'] + line.start.ordinal and \
                  l['end_time'] <= file['parses']['start_ms'] + line.end.ordinal:
                    interjection_lines.remove(l)
                    if not is_a_repeat(line_record, l['raw_line']) and len(l['raw_line'].strip()) > 1:
                      line_record.append(l['raw_line'])
                      out.write(l['line'])
                      show_original_speaker_after_interjection = True
    if len(subs) > 0 and len(subs[0].text.strip()) > 1:
      out.write('\n')
    #out.write(str(convert_ms_to_hms(info['end_ms'])))
    