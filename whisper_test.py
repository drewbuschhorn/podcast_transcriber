import whisper 
import torch
from whisper.utils import get_writer

# specify the path to the input audio file
input_file = "episode-24b.mp3"

# specify the path to the output transcript file
output_file = "transcript.txt"

# Cuda allows for the GPU to be used which is more optimized than the cpu
torch.cuda.init()
device = "cuda" # if torch.cuda.is_available() else "cpu"
print ("Using device:", device)

#load whisper model
model_size = "large-v2"
print("loading model :", model_size)
model = whisper.load_model(model_size).to(device)
print(model_size, "model loaded")

# Initialize variables

# Transcribe audio
with torch.cuda.device(device):
    result = model.transcribe(input_file, fp16=False)
    writer = get_writer('srt', 'output')
    writer(result, output_file)
    print (result)