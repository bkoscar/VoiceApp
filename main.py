from encoder.params_model import model_embedding_size as speaker_embedding_size
# from utils.argutils import print_args
# from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf 
import librosa
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError


# # English models
# model_path = "/home/oscar/Documents/AudioCloning/VoiceApp/modelos/english/"
# encoder_path = model_path + "encoder.pt"
# syn_path = model_path + "synthesizer.pt"
# voc_path = model_path + "vocoder.pt"

#Spanish models
model_path = "/home/oscar/Documents/AudioCloning/VoiceApp/modelos/pretrained_spanish/"
encoder_path = model_path + "encoder/saved_models/pretrained.pt"
syn_path = model_path + "synthesizer/saved_models/pretrained/pretrained.pt"
voc_path = model_path + "/vocoder/saved_models/pretrained/pretrained.pt"


encoder.load_model(Path(encoder_path))
synthesizer = Synthesizer(Path(syn_path))
vocoder.load_model(Path(voc_path))


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponible")
    print("Nombre del GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU no disponible, utilizando CPU")

print("Versión de PyTorch:", torch.__version__)
print("Dispositivo actual:", device)
