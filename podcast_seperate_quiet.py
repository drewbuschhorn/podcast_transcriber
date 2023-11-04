import os
from collections import defaultdict
from pydub import AudioSegment
import argparse
import logging
import math
from tqdm import tqdm

def parse_filename(filename):
    start_ms_str, rest = filename.split('ms-', 1)
    start_ms = int(start_ms_str)
    
    end_ms_str, rest = rest.split('ms_speaker_', 1)
    end_ms = int(end_ms_str)
    
    speaker = rest.split('_')[1]
    return start_ms, end_ms, speaker

def combine_files(input_file):
    directory, mp3_filename = os.path.split(input_file)
    base_filename = mp3_filename
    output_directory = os.path.join(directory, "output", base_filename.replace(".", "_").replace("-","_"))
    
    files = [f for f in os.listdir(output_directory) if f.endswith('.mp3') and not f.startswith("combined")]
    
    speaker_files = defaultdict(list)
    max_end_ms = 0
    all_segments = []

    for f in files:
        start_ms, end_ms, speaker = parse_filename(f)
        speaker_files[speaker].append(f)
        all_segments.append((start_ms, end_ms, speaker))
        max_end_ms = max(max_end_ms, end_ms)

    # Identify overlapping segments
    overlaps = defaultdict(list)
    all_segments = sorted(all_segments, key=lambda x: x[0])

    for i, (start1, end1, speaker1) in enumerate(all_segments):
        for start2, end2, speaker2 in all_segments[i+1:]:
            if start2 < end1:  # Overlap detected
                overlaps[(start1, end1, speaker1)].append((start2, end2, speaker2))
            else:
                break

    for speaker, s_files in speaker_files.items():
        files_sorted = sorted(s_files, key=lambda x: parse_filename(x)[0])
        combined = AudioSegment.silent(duration=max_end_ms)
        
        for f in tqdm(files_sorted, desc=f"Processing files for speaker {speaker}", ncols=100, unit="file"):
            start_ms, end_ms, _ = parse_filename(f)
            current_segment = AudioSegment.from_mp3(os.path.join(output_directory, f))

            # Adjust volume for overlaps
            num_overlaps = len(overlaps.get((start_ms, end_ms, speaker), [])) + 1  # including the segment itself
            adjusted_segment = current_segment - (20 * math.log10(math.sqrt(num_overlaps)))
            
            # Overlay the adjusted segment on the base track
            combined = combined.overlay(adjusted_segment, position=start_ms)

        combined.export(os.path.join(output_directory, f"combined_{speaker}.mp3"), format="mp3")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='combine_speaker_files',
        description='Combines MP3 files by speaker and adjusts overlaps',
        epilog='Ensure that the input files are named correctly')
    
    parser.add_argument('--file', metavar='-f', required=True,
                        help='input MP3 file path to process')
    parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    combine_files(args.file)
