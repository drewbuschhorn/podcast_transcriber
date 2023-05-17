# Installation

1. Install Python 3.10 (required currently for pyannote, python 3.11 will not work).
1. Create a virtual environment with `python -m venv venv` and activate it (call varies by platform).
1. Verify you're in a Python 3.10 environment, `python --version`.
1. Get an API account with OpenAI, and make an API key. https://platform.openai.com/signup (No payment needed initially, about ~0.20c per podcast if all steps taken)
1. Get an API account with HuggingFace and make an API key. https://huggingface.co/welcome (No payment)
1. Cp .env.template to .env and set your API tokens.
1. Follow https://github.com/pyannote/pyannote-audio#tldr- but skip the dev branch install (requirements.txt will handle that). This will have you activate some license agreements in HuggingFace.
1. Run `pip install --upgrade pip` then `pip3 install -r requirements.txt`, this will take at least 1 GB of space of libraries.
1. Assuming all of the above worked, verify you have CUDA available (nvidia gpu library), and that pytorch with CUDA has been installed. (Run `python torch_test.py` if you're unsure.)

# Actually run the thing
1. Save an mp3 to a directory. Downsample it to a constant-bit-rate and mono-audio.
1. Run `python podcast_diarization_to_tracks.py --file {mp3_filename} --num_speakers {best_guess_number_of_voices_in_podcast}`. This will create a large number of files in the output directory named after the mp3 file.
1. Run `python podcast_tracks_to_srts.py --file {mp3_filename}` use same filename as above. This will send the files generated above to OpenAI to be converted into 'srt' (subtitle format) files in the output directory. THIS COSTS MONEY (about $0.05 for half-hours audio), and is a bit slow.
1. Run `python srts_to_transcript.py --file {mp3_filename}` to convert those srt files to a single unified transcript with interjections. If you examine the file and can easily tell which SPEAKER_XX is which, you can specify a rerun like `python srts_to_transcript.py --file {mp3_filename} --names [{\"name\":\"Drew\",\"speaker_id\":\"SPEAKER_00\"},{\"name\":\"Kristin\",\"speaker_id\":\"SPEAKER_01\"}]` which will just do a dumb find-replace. The output will be in the output directory in the path {mp3_filename}\real.txt
1. (Optional) Run `python podcast_transcript_gpt_fixup.py --file {mp3_filename}` which will send the {mp3_filename}\real.txt file to OpenAI GPT3 in chunks. THIS COSTS MONEY (about $~0.10 per file). Output will be in {mp3_filename}\fixup-{timestamp}.txt. **I'm going to overhaul this step later. I'm not happy with it. Maybe use langchain and a summarization system. idk.**