# 🧉 milei-gpt

Che y si queremos hacer un LLM que hable de la misma forma que un famoso ... como hacemos? Este repo es una excusa para aprender como preparar un dataset para fine-tuning de algún LLM, como evaluarlo, como tokenizarlo, como extenderlo de formar sintética, y tantas otras cosas. Al final, vamos a tener un modelo que (si todo sale bien) va a hablar como la persona que elegimos. Por ahora, construido sobre Llama3-8B, y usando APIs públicas para procesar la data, sobre mas de 70 horas de entrevistas.

## Paso a paso, que vamos a hacer
- Encontrar todas las entrevistas en YT de algún famoso/a
- Transcribir las entrevistas
- Preparar un dataset (convertir a `ChatML`, tokenization, data sintética, etc.)
- Elegir un modelo base eg. `Llama3-8B` o `Phi-3-mini-128k-instruct`
- Fine-tuning del LLM
- Evaluación del modelo y push to HF

## Links para ir leyendo y tener en cuenta
- https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
- https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4
- https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d
- https://github.com/e-p-armstrong/augmentoolkit
- https://huggingface.co/blog/burtenshaw/domain-specific-datasets
- https://huggingface.co/spaces/argilla/domain-specific-datasets-welcome
- https://www.reddit.com/r/LocalLLaMA/

## 🛠 Buenos, vamos a arrancar, que necesitás?

- Una GPU con >= 8GB de VRAM (local o [Google Colab](https://colab.research.google.com/), [Sagemaker StudioLab](https://studiolab.sagemaker.aws/), o algo como [LightingAI](https://lightning.ai/)).
- `Python>=3.10`
- `yt-dlp`
- `whisperX` para transcribir el audio.
- `pyannote-audio` para reconocer las voces y diferenciar a los hablantes.
- `NVIDIA NeMo` para diarización de audio.

### Cloná el repo y armá el environment

```bash
$ git clone https://github.com/machinelearnear/milei-gpt
$ cd milei-gpt
```

Podés instalar las librerias con `pip install -r requirements.txt` o directamente con el `environment.yml`.

```
$ conda env create -f environment.yml
$ conda activate machinelearnear-dev
```

### Vamos a ver el código un poco
Acá tenés los dos scripts principales: uno para procesar datos de YouTube y otro para transcribir y diarizar el audio.

**`0_encontrar_raw_data_en_youtube.py`**

Este script se encarga de extraer información de YouTube sobre los videos relevantes y te lo baja a `./data/dataset.csv`.

```
# ejecutar el script para extraer datos de YouTube
python 0_encontrar_raw_data_en_youtube.py --data_dir_hq ./data/youtube_data_hq --data_dir_lq ./data/youtube_data_lq --output_filename ./data/dataset.csv

```

**`b_whisper_transcription_nemo_diarization.py`**

Este script realiza la transcripción y diarización de los audios descargados, generando archivos JSON con los segmentos transcritos y anotados por hablante.

```
# ejecutar el script para transcribir y diarizar
python b_whisper_transcription_nemo_diarization.py --hf_token TU_HF_TOKEN --input_file ./data/dataset.csv --ref_audio_dir ./data/reference_audio --output_dir ./output
```

## 💡 Algunas notas

- Seguí los pasos como te los di, así no hay lío.
- Si te encontrás con algún error, revisá que las rutas de los archivos estén bien y que hayas puesto todos los argumentos necesarios.
- No te olvides de poner tu token de Hugging Face cuando uses el script.
- Contar tokens para Llama-2: https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/ y para OpenAI: https://platform.openai.com/tokenizer

## 🤝 ¿Querés sumarte?

Si te gusta lo que hicimos y tenés alguna idea para mejorarlo, ¡dale, unite! Mandá lo que hiciste y vemos cómo lo metemos.

## 📬 ¿Dudas?

Si algo no te cierra o necesitás una mano, escribinos. Estamos para ayudarte.