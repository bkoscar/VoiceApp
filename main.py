# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
# from pathlib import Path
# import numpy as np
# import soundfile as sf 
# import librosa
# import argparse
# import torch
# import sys
# import os
# from audioread.exceptions import NoBackendError

# def main():
#     parser = argparse.ArgumentParser(description='Voice Cloning')
#     parser.add_argument('--text', type=str, help='Text to synthesize')
#     parser.add_argument('--audio', type=str, help='Path to input audio file')
#     args = parser.parse_args()

#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         ## Print some environment information (for debugging purposes)
#         print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#             "%.1fGb total memory.\n" %
#             (torch.cuda.device_count(),
#             device_id,
#             gpu_properties.name,
#             gpu_properties.major,
#             gpu_properties.minor,
#             gpu_properties.total_memory / 1e9))
#     else:
#         print("Using CPU for inference.\n")
    
#     # Check if text and audio arguments are provided
#     if args.text is None or args.audio is None:
#         print("Please provide both text and audio file arguments.")
#         return

#     # Load pretrained models
#     model_path = "/home/oscar/Documents/AudioCloning/VoiceApp/models/spanish/pretrained_spanish/"
#     encoder_path = model_path + "encoder/saved_models/pretrained.pt"
#     syn_path = model_path + "synthesizer/saved_models/pretrained/pretrained.pt"
#     voc_path = model_path + "vocoder/saved_models/pretrained/pretrained.pt"

#     encoder.load_model(Path(encoder_path))
#     synthesizer = Synthesizer(Path(syn_path))
#     vocoder.load_model(Path(voc_path))

#     # Load input audio file
#     in_fpath = Path(args.audio)
#     print("Input audio file:", in_fpath)
#     original_wav, sampling_rate = librosa.load(str(in_fpath))
#     preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#     print("Loaded audio file successfully")

#     # Preprocess text and synthesize spectrogram
#     text = args.text
#     print("Text to synthesize:", text)
#     texts = [text]
#     embeds = [encoder.embed_utterance(preprocessed_wav)]
#     specs = synthesizer.synthesize_spectrograms(texts, embeds)
#     spec = specs[0]
#     print("Created the mel spectrogram")

#     # Synthesize waveform
#     print("Synthesizing the waveform:")
#     generated_wav = vocoder.infer_waveform(spec)
#     generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#     generated_wav = encoder.preprocess_wav(generated_wav)

#     # Save output waveform
#     output_filename = "demo_output.wav"
#     sf.write(output_filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
#     print("Saved output as", output_filename)

# if __name__ == "__main__":
#     main()
from encoder.params_model import model_embedding_size as speaker_embedding_size
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
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Clonación de voz')
    parser.add_argument('--text', type=str, help='Texto a sintetizar')
    parser.add_argument('--audio', type=str, help='Ruta al archivo de audio de entrada')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Imprimir información del entorno (para propósitos de depuración)
        print("Se encontraron %d GPUs disponibles. Usando la GPU %d (%s) con capacidad de cálculo %d.%d con "
            "%.1fGb de memoria total.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Usando CPU para inferencia.\n")
    
    # Verificar si se proporcionan los argumentos de texto y audio
    if args.text is None or args.audio is None:
        print("Por favor, proporcione tanto el texto como el archivo de audio.")
        return

    # Cargar modelos preentrenados
    model_path = "/home/oscar/Documents/AudioCloning/VoiceApp/models/spanish/pretrained_spanish/"
    encoder_path = model_path + "encoder/saved_models/pretrained.pt"
    syn_path = model_path + "synthesizer/saved_models/pretrained/pretrained.pt"
    voc_path = model_path + "vocoder/saved_models/pretrained/pretrained.pt"

    encoder.load_model(Path(encoder_path))
    synthesizer = Synthesizer(Path(syn_path))
    vocoder.load_model(Path(voc_path))

    # Cargar archivo de audio de entrada
    in_fpath = Path(args.audio)
    print("Archivo de audio de entrada:", in_fpath)
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Archivo de audio cargado exitosamente")

    # Preprocesar texto y sintetizar espectrograma
    text = args.text
    print("Texto a sintetizar:", text)
    texts = [text]
    embeds = [encoder.embed_utterance(preprocessed_wav)]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Se creó el espectrograma mel")

    # Sintetizar forma de onda
    print("Sintetizando la forma de onda:")
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Obtener la fecha y hora actual
    current_time = datetime.now()

    # Formatear la fecha y hora actual como una cadena
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Nombre del archivo de salida con la fecha y hora actual
    output_filename = "demo_output_" + time_str + ".wav"

    # Guardar forma de onda de salida
    sf.write(output_filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("Se guardó la salida como", output_filename)

if __name__ == "__main__":
    main()
