import os
from collections import defaultdict
from pydub import AudioSegment
import argparse
import logging
from tqdm import tqdm

def parse_filename(filename):
    start_ms_str, rest = filename.split('ms-', 1)
    start_ms = int(start_ms_str)
    
    end_ms_str, rest = rest.split('ms_speaker_', 1)
    end_ms = int(end_ms_str)
    
    speaker = rest.split('_')[1]  # Extracting SPEAKER from 'SPEAKER_00.mp3'

    return start_ms, end_ms, speaker

def combine_files(input_file):
    directory, mp3_filename = os.path.split(input_file)
    base_filename = mp3_filename
    output_directory = os.path.join(directory, "output", base_filename.replace(".", "_").replace("-","_"))
    
    # Get all mp3 files in the directory
    files = [f for f in os.listdir(output_directory) if f.endswith('.mp3') and not f.startswith("combined")]
    
    # Group files by speaker
    speaker_files = defaultdict(list)
    max_end_ms = 0

    for f in files:
        start_ms, end_ms, speaker = parse_filename(f)
        speaker_files[speaker].append(f)
        max_end_ms = max(max_end_ms, end_ms)

    for speaker, s_files in speaker_files.items():
        # Sort files by start time for each speaker
        files_sorted = sorted(s_files, key=lambda x: parse_filename(x)[0])
        
        # Create a silent base track for the full duration
        combined = AudioSegment.silent(duration=max_end_ms)
        
        for f in tqdm(files_sorted, desc=f"Processing files for speaker {speaker}", ncols=100, unit="file"):
            start_ms, end_ms, _ = parse_filename(f)
            current_segment = AudioSegment.from_mp3(os.path.join(output_directory, f))
            
            # Overlay the current segment on the base track
            combined = combined.overlay(current_segment, position=start_ms)

        # Save the combined audio for each speaker in the output_directory
        combined.export(os.path.join(output_directory, f"combined_{speaker}.mp3"), format="mp3")
        
# Use the function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='combine_speaker_files',
        description='Combines MP3 files by speaker based on file naming conventions',
        epilog='Ensure that the input files are named correctly')
    
    parser.add_argument('--file', metavar='-f', required=True,
                        help='input MP3 file path to process')
    parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    combine_files(args.file)
