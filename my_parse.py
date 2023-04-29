# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline
import csv
from pyannote.audio import Pipeline
from pydub import AudioSegment
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                                        use_auth_token="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")
                                                                        

# 4. apply pretrained pipeline
pipeline = pipeline.to(0)
diarization = pipeline("input/episode-10.mp3", num_speakers=2)
audio_file = AudioSegment.from_file("input/episode-10.mp3")


# 5. print the result
audio_snippets = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)
    duration = end_ms - start_ms
    filename = f"output/{start_ms}ms-{end_ms}ms_speaker_{speaker}.mp3"
    print (filename)
    audio_snippet = audio_file[start_ms:end_ms]
    audio_snippet.export(filename, format="mp3")
    audio_snippets.append((duration, speaker, start_ms, end_ms))


with open("output/durations_and_speakers.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Duration (s)", "Speaker", "start_ms", "end_ms"])
    for duration, speaker, start_ms, end_ms in audio_snippets:
        writer.writerow([duration, speaker, start_ms, end_ms])