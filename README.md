# 🧉 milei-gpt

Che y si queremos hacer un LLM que hable de la misma forma que un famoso ... como hacemos? Este repo es una excusa para aprender a preparar un dataset para fine-tunear algún LLM, aprender como evaluarlo, como tokenizarlo, como extenderlo de formar sintética, y tantas otras cosas. Al final, si todo sale bien, vamos a tener un modelo que va a hablar como la persona que elegimos, y le podemos poner un RAG (retrieval augmented generation) encima para que nos traiga un contexto correcto y factual en las respuestas. Por ahora, la idea es hacerlo sobre Llama3-8B y usando APIs públicas para procesar la data, sobre mas de 100 horas de entrevistas.

## Paso a paso, que vamos a hacer
- Encontrar todas las entrevistas en YT de algún famoso/a
- Transcribir las entrevistas
- Preparar un dataset (convertir a `ChatML`, tokenization, data sintética, etc.)
- Elegir un modelo base eg. `Llama3-8B` o `Phi-3-mini-128k-instruct`
- Fine-tuning del LLM, evaluación del modelo, y push to HF.
- Armar un RAG indexando las entrevistas y meterle este LLM

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

Armá todo directando con el `environment.yml` o sino instalá las librerias con `pip install -r requirements.txt`.

```
$ conda env create -f environment.yml
$ conda activate machinelearnear-dev
```

### Vamos a ver el código un poco
Como todo esto es un WIP (work in progress) lo tengo todo puesto adentro de `./scripts`. Vamos a ver algunos: 

**`0_encontrar_raw_data_en_youtube.py`**

Este script se utiliza para extraer información de videos de YouTube de ciertos candidatos, procesar esa información para obtener detalles específicos sobre los videos (como duración, vistas, etc.), y luego filtrar y guardar los datos en `./data/dataset.csv`. Además, el script organiza los datos tanto de videos de alta calidad (HQ) como de baja calidad (LQ) para su posterior uso.

```
python 0_encontrar_raw_data_en_youtube.py --data_dir_hq ./data/youtube_data_hq \
    --data_dir_lq ./data/youtube_data_lq --output_filename ./data/dataset.csv
```

**`1_whisper_transcription_nemo_diarization.py`**


El script descarga audios de YouTube que teniamos guardados en `./data/dataset.csv`, realiza transcripciones utilizando Whisper y realiza la diarización de los segmentos de audio para identificar y separar las diferentes voces en el audio usando NeMo. Estoy usando una versión modificada de este repo: [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization). Finalmente, guarda los resultados en archivos JSON en `./output/{target_speaker_name}/{youtube_video_id}.json`.

Esta es la estructura de los JSON que vamos a tener de output, eg. `./output/milei/ad5K-6f8dqs.json`, pueden ver las conversaciones mergeadas dentro de los segmentos (varias por el mismo speaker).

**Top-level keys:** These are metadata fields providing information about the video.

- "Index": An integer index.
- "channel_id": The ID of the YouTube channel.
- "channel": The name of the channel.
- "uploader_url": The URL of the uploader's YouTube page.
- "id": The ID of the video.
- "url": The URL of the video.
- "title": The title of the video.
- "duration": The duration of the video in seconds.
- "view_count": The view count of the video.
- "candidate_name": The name of the candidate.
- "quality": The quality of the video.

**Segments:** An array of dictionaries, each representing a segment of the video with the following keys:

- "speaker": The speaker of the segment.
- "start_time": The start time of the segment in seconds.
- "end_time": The end time of the segment in seconds.
- "text": The text spoken during the segment.
- "cosine_dist": The cosine distance value (probably used for speaker verification).
- "is_target_speaker": A boolean indicating if the speaker is the target speaker.

```json
{
    "Index": 67,
    "channel_id": "UCz489cQmrgH57sShDiatwfw",
    "channel": "El Peluca Milei",
    "uploader_url": "https://www.youtube.com/@ElPelucaMilei",
    "id": "ad5K-6f8dqs",
    "url": "https://www.youtube.com/watch?v=ad5K-6f8dqs",
    "title": "LA ÚLTIMA ENTREVISTA DE MILEI ANTES DE SER PRESIDENTE",
    "duration": 1149.0,
    "view_count": 1206055,
    "candidate_name": "milei",
    "quality": "high",
    "segments": [
        {
            "speaker": "Speaker 1",
            "start_time": 0.62,
            "end_time": 50.614,
            "text": "¿Seguís el dólar?  No.  Balotage.  ¿Te imaginás en él?  Y probamos a ser todos más pobres.  Tengo una consulta sencilla.  En el país de mi ley, yo voy, compro un arma, ¿la puedo llevar?  Es falso.  Nunca dijiste, no se cometieron delitos de lesa humanidad.  ¿Es que hay corrupción en el INCUCAI?  En Argentina hay como un consenso de que tengas la guita que tengas, tenés que ir a una lista.  ¿Realmente creés que hay corrupción?  Tenés cara de... Te vas a sorprender.  el domingo.  Fue el más votado en las PASO y es el que hizo las propuestas más disruptivas en la campaña.  Algunas realmente polémicas.  Le pedimos en el Movistar Arena, a minutos del cierre de campaña, quince minutos a solas, su camarín.  A Javier Milei.  Javier. ",
            "cosine_dist": 0.7135412962560479,
            "is_target_speaker": false
        },
        {
            "speaker": "Speaker 0",
            "start_time": 51.214,
            "end_time": 52.235,
            "text": "¿Qué tal, Rodolfo? ",
            "cosine_dist": 0.8585994301094911,
            "is_target_speaker": false
        },
        {
            "speaker": "Speaker 1",
            "start_time": 52.275,
            "end_time": 53.756,
            "text": "Gracias por recibirnos. ",
            "cosine_dist": 0.8117557859327827,
            "is_target_speaker": false
        },
        ...
    ]
}
```

Y asi lo ejecutamos

```
python b_whisper_transcription_nemo_diarization.py --hf_token TU_HF_TOKEN \
    --input_file ./data/dataset.csv --ref_audio_dir ./data/reference_audio --output_dir ./output
```

Aca tienen los pasos donde se hace mas o menos todo:

```python
def main():
    parser = argparse.ArgumentParser(description='transcribe & diarize speech segments')
    ...
    if os.path.exists(f'{output_dir}/{each.candidate_name}/{each.id}.json'):
        logger.info(f'File already exists!')
        continue

    # start audio pipeline
    io = Audio(sample_rate=16000, mono="downmix")
    ref_audio_filename = f'{ref_audio_dir}/audio_{each.candidate_name}.wav'
    reference_segment = Segment(1., 10.)
    reference_embeddings = inference_pipeline.crop(ref_audio_filename, reference_segment)

    try:
        download_audio(each.url, temp_dir)
    except Exception as e:
        logger.error(f"Failed to download audio for: {each.url}. Error: {e}")
        continue

    is_main_speaker, appearance_time_from_main_speaker = check_if_main_speaker_or_not(io, inference_pipeline, reference_embeddings, audio_path, cdist_threshold, embedding_model, min_appearance_time)
    ...
    if not is_main_speaker:
        logger.info(f'Speaker "{each.candidate_name}" was not found')
        shutil.rmtree(temp_dir)
        continue
    ...
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
    msdd_model.diarize()
    del msdd_model
    torch.cuda.empty_cache()

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        ...
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if language in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list)
        ...
        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            ...
            word_dict["word"] = word
    else:
        logger.warning(f"Punctuation restoration is not available for {language} language. Using the original punctuation.")

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    logger.info(f'Finished transcription with Whisper')
    waveform_target, sample_rate_target = torchaudio.load(audio_path)
    wav_duration = waveform_target.shape[1] / sample_rate_target

    logger.info(f'Speaker verification for each audio segment...')
    merged_segments = merge_segments(ssm)

    for chunk in merged_segments:
        start_time = chunk['start_time'] / 1000.0
        end_time = chunk['end_time'] / 1000.0
        ...
        if (end_time - start_time) <= 0.6:
            chunk['cosine_dist'] = 1.0
            chunk['is_target_speaker'] = False
            continue

        segment_chunk = Segment(start_time, end_time)
        waveform_chunk, sample_rate = io.crop(audio_path, segment_chunk)
        chunk_embeddings = inference_pipeline({"waveform": waveform_chunk, "sample_rate": sample_rate})
        distance = cdist(np.reshape(reference_embeddings, (1, -1)), np.reshape(chunk_embeddings, (1, -1)), metric="cosine")
        chunk['start_time'] = start_time
        chunk['end_time'] = end_time
        chunk['cosine_dist'] = float(distance[0][0])
        chunk['is_target_speaker'] = bool(distance[0][0] <= cdist_threshold)

    logger.info(f'Finished! ...')
    save_dict_to_file(merged_segments, each, each.candidate_name, output_dir, filename=each.id)
    cleanup(temp_path)

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