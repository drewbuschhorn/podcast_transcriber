# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline
import logging
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                                        use_auth_token="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")

# 4. apply pretrained pipeline
diarization = pipeline("untitled.wav")

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
        logging.debug(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        # start=0.2s stop=1.5s speaker_0
        # start=1.8s stop=3.9s speaker_1
        # start=4.2s stop=5.7s speaker_0
        # ...
