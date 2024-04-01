# VoiceApp
Este proyecto presenta una aplicación en Python diseñada para clonar voces utilizando la potencia de PyTorch. Con soporte para los idiomas inglés y español, la aplicación emplea una variedad de modelos para codificar, sintetizar y vocodear el audio.

## Funcionalidades Principales:
 1. Clonación de Voces: La aplicación permite al usuario clonar voces utilizando técnicas avanzadas de procesamiento de señales de audio.
 2. Entrada de Texto y Audio: Los usuarios pueden ingresar un texto junto con un fragmento de audio de aproximadamente 5 segundos, y la aplicación generará un audio clonado con el texto proporcionado.

## Requerimientos
- Python 3.9
- FFmpeg
- [PyTorch](https://pytorch.org/)

## Modelos
Descarga la carpeta [aquí](https://drive.google.com/drive/folders/1E3-rXgiX0VbfVpQb7QUi85zGtQMeYRBk?usp=sharing) que contiene los modelos.

## Instalación y Uso
Para instalar los requerimientos, ejecuta el siguiente comando:
```bash
pip install -r requerimientos.txt
```



# Correr la inferencia
```bash
python main.py --text "<text>" --audio "<audio_de_referencia>"
```

# Nota
En el archivo main.py  colocar las rutas de los modelos
```python
    model_path = "/models/spanish/pretrained_spanish/"
    encoder_path = model_path + "encoder/saved_models/pretrained.pt"
    syn_path = model_path + "synthesizer/saved_models/pretrained/pretrained.pt"
    voc_path = model_path + "vocoder/saved_models/pretrained/pretrained.pt"
```