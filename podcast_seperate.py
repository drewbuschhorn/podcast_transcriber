import os
from collections import defaultdict
from pydub import AudioSegment

def parse_filename(filename):
    start_ms_str, rest = filename.split('ms-', 1)
    start_ms = int(start_ms_str)
    
    end_ms_str, rest = rest.split('ms_speaker_', 1)
    end_ms = int(end_ms_str)
    
    speaker = rest.split('_')[1]  # Extracting SPEAKER from 'SPEAKER_00.mp3'

    return start_ms, end_ms, speaker

def combine_files(directory):
    # Get all mp3 files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    
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
        
        for f in files_sorted:
            start_ms, end_ms, _ = parse_filename(f)
            current_segment = AudioSegment.from_mp3(os.path.join(directory, f))
            
            # Overlay the current segment on the base track
            combined = combined.overlay(current_segment, position=start_ms)

        # Save the combined audio for each speaker
        combined.export(os.path.join(directory, f"combined_{speaker}"), format="mp3")
        
# Use the function
combine_files("./output/episode_23_mp3/")
