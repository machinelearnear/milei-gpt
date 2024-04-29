import pandas as pd
import numpy as np
import os
import yt_dlp
import argparse
import whisperx
import json
import shutil
import math
import logging
import torch
import gc

from tqdm import tqdm
from pathlib import Path
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist

# ---- extra vocab ---- #
prompt = "LELIQ FMI CONICET AFIP PBI AFJP Milei Bullrich Massa Bregman Schiaretti"

# ---- logging ---- #
# set up root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
log_file = './logs/run_transcription.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# create a stream handler (for writing logs to console)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# formatter for the logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ---- helper funcs ---- #
def download_audio(URL, output_dir, audio_fname:str = 'audio'):
    """
    Download the audio from a YouTube video as a WAV file.
    Args:
        URL (str): The URL of the YouTube video.
        output_dir (str): The directory to save the output file in.
        audio_filename (str): The filename to save the output file as.
    """
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, audio_fname),
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(URL)

def save_dict_to_disk(data_dict, data, candidate_name, save_dir, filename:str = None):
    """
    Save a dictionary to disk as a JSON file.
    Args:
        data_dict (dict): The dictionary to save.
        data (namedtuple): A named tuple representing a row from a dataframe.
        name_reference (dict): A dictionary mapping channel names to some values.
        filename (str): The filename to save the output file as (without extension).
    """

    def _process_dictionary(data_dict):
        """Process the dictionary to remove specified keys and replace NaN values."""
        # Remove specified keys
        if "Index" in data_dict:
            del data_dict["Index"]
        if "word_segments" in data_dict:
            del data_dict["word_segments"]
        for segment in data_dict.get("segments", []):
            if "words" in segment:
                del segment["words"]

        # recursively replace NaN values with None
        def replace_nan_with_null(data):
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (list, dict)):
                        replace_nan_with_null(item)
                    elif isinstance(item, float) and math.isnan(item):
                        data[i] = None
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (list, dict)):
                        replace_nan_with_null(value)
                    elif isinstance(value, float) and math.isnan(value):
                        data[key] = None
            return data

        return replace_nan_with_null(data_dict)

    # process the data dictionary
    processed_data_dict = _process_dictionary(data_dict)

    output_dir = f"{save_dir}/{candidate_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.json")

    # convert the named tuple to a dictionary
    row_dict = data._asdict()

    # merge the two dictionaries
    combined_dict = {**processed_data_dict, **row_dict}

    with open(filepath, 'w') as f:
        json.dump(combined_dict, f)

    logging.info(f'File saved to: "{filepath}"')

def verify_speaker_in_audio(
        audio_pipe, reference_wav_fname, segment_reference,
        waveform_reference, embedding_reference, target_audio,
        cdist_threshold, embeddings_model, min_appearance_time):

    # load the raw audio file
    waveform, sample_rate = audio_pipe(target_audio)

    # determine the total duration of the audio file
    total_duration = waveform.shape[1] / sample_rate
    num_chunks = int(np.ceil(total_duration / 30))
    results = []
    passed_check = False

    # split audio into 30-second chunks and perform speaker verification for each chunk at diff. checks
    for i in range(num_chunks):
        start_time = i * 30
        end_time = (i + 1) * 30

        # handle the last segment which might be less than 30s
        if end_time > total_duration:
            end_time = total_duration

        # extract the current 30-second chunk
        segment_target = Segment(start_time, end_time)
        waveform_target, _ = audio_pipe.crop(target_audio, segment_target)

        # extract embedding for the chunk
        embedding_target = embeddings_model(waveform_target[None])

        # compare embeddings using "cosine" distance
        distance = cdist(embedding_reference, embedding_target, metric="cosine")

        chunk_result = {
            'start_time': start_time,
            'end_time': end_time,
            'cosine_dist': distance[0][0],
            'is_candidate': True if distance[0][0] <= cdist_threshold else False
        }
        results.append(chunk_result)

        if i == 10:  # or 5 minutes, depending on your chunk size
            num_true_candidates = sum([result['is_candidate'] for result in results])
            early_check = num_true_candidates / len(results)
            if early_check >= min_appearance_time:
                passed_check = True
                break

        elif i == int(num_chunks * 0.50):
            num_true_candidates = sum([result['is_candidate'] for result in results])
            mid_check = num_true_candidates / len(results)
            if mid_check >= min_appearance_time:
                passed_check = True
                break

        if not passed_check and i > int(num_chunks * 0.50):
            break  # exit loop if no checks passed and we're past this point

    num_true_candidates = sum([result['is_candidate'] for result in results])
    return passed_check, ( num_true_candidates / len(results) )

def main():
    parser = argparse.ArgumentParser(description='process and transcribe YouTube videos.')
    parser.add_argument('--hf_token', help='Hugging Face Token')
    parser.add_argument('--cdist_threshold', type=float, default=0.5,
                        help='Threshold for cdist (default: 0.5)')
    parser.add_argument('--input_file', default='../data/dataset.csv',
                        help='File containing youtube URLs (default: ../data/dataset.csv)')
    parser.add_argument('--ref_audio_dir', default='../data/reference_audio',
                        help='Reference audio directory (default: ../data/reference_audio)')
    parser.add_argument('--temp_dir', default='../data/temp_audio',
                        help='Temporary directory for intermediate files (default: ../data/temp_audio)')
    parser.add_argument('--output_dir', default='../output',
                        help='Output directory for saving results (default: ../output)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--compute_type', default="float16",
                        help='Compute type for processing (default: float16)')
    parser.add_argument('--min_appearance_time', default=0.10,
                        help='Percentage time we want the main speaker to speak in the video')
    args = parser.parse_args()

    # setting device on GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')
    
    # assigning variables from argparse
    hf_token = args.hf_token
    cdist_threshold = args.cdist_threshold
    input_file = args.input_file
    ref_audio_dir = args.ref_audio_dir
    temp_dir = args.temp_dir
    output_dir = args.output_dir
    target_audio = f"{args.temp_dir}/audio.wav"
    batch_size = args.batch_size
    compute_type = args.compute_type
    min_appearance_time = args.min_appearance_time

    # load whisper model
    vad_opts = {'vad_onset': 0.1, 'vad_offset': 0.1}
    model = whisperx.load_model(
        "large-v3", device=device, compute_type=compute_type, vad_options=vad_opts)
    logging.info('OpenAI whisper "large-v2" model is loaded')

    # load embeddings model
    embeddings_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        use_auth_token=hf_token,
        device=torch.device("cuda"))
    logging.info('Embeddings model "speechbrain/spkrec-ecapa-voxceleb" loaded')

    # read data and start pipeline
    data = pd.read_csv(input_file)
    for each in tqdm(data.itertuples(), total=data.shape[0]):
        # check if file was already processed
        logging.info('\n---------\n')
        if os.path.exists(f'{output_dir}/{each.candidate_name}/{each.id}.json'):
            # print('\033[1m' + f'skipping: {each.title}' + '\033[0m')
            logging.info(f'File already exists! Video: "{each.title}"')
            continue
        else:
            logging.info('\033[1m'+f'Processing video: "{each.title}"'+'\033[0m')

        # start audio pipeline
        audio_pipe = Audio(sample_rate=16000, mono="downmix")
        reference_wav_fname = f'{ref_audio_dir}/audio_{each.candidate_name}.wav'

        # extract embeddings for a reference (candidate) speaker speaking between t=0 and t=10s
        segment_reference = Segment(1., 10.)
        waveform_reference, sample_rate = audio_pipe.crop(
            reference_wav_fname, segment_reference)
        embedding_reference = embeddings_model(waveform_reference[None])

        # download audio from youtube url
        try:
            download_audio(each.url, temp_dir)
        except Exception as e:
            logging.error(f"Failed to download audio for URL {each.url}. Error: {e}")
            continue

        # double-check if candidate is the main speaker in the video or not
        logging.info(f'Checking if candidate "{each.candidate_name}" is the main speaker..')
        is_main_speaker, appearance_time_from_main_speaker = verify_speaker_in_audio(
            audio_pipe, reference_wav_fname, segment_reference,
            waveform_reference, embedding_reference, target_audio,
            cdist_threshold, embeddings_model, min_appearance_time)

        if not is_main_speaker:
            logging.info(f'Candidate "{each.candidate_name}" not found')
            logging.info(f'Candidate only appeared: {appearance_time_from_main_speaker:.2f}%')
            shutil.rmtree(temp_dir)
            continue
        else:
            logging.info(f'Candidate "{each.candidate_name}" WAS found!')

        # ---- transcription starts ---- #
        logging.info(f'Starting transcription for: "{each.url}"')
        # transcribe and perform speaker diarization
        audio = whisperx.load_audio(target_audio)
        result = model.transcribe(
            audio, batch_size=batch_size, language='es')

        # align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio,
            device, return_char_alignments=False)
        gc.collect(); torch.cuda.empty_cache(); del model_a # delete model if low on GPU resources

        # assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logging.info(f'Finished transcription for: "{each.url}"')

        # ---- speaker verification starts ---- #
        # load the raw audio file
        waveform, sample_rate = audio_pipe(target_audio)
        # determine the total duration of the audio file
        wav_duration = waveform.shape[1] / sample_rate

        # add cosine dist to each segment in the results
        for segment in result['segments']:
            # extract embedding for a speaker speaking between t=Xs and t=Ys
            if segment['end'] > wav_duration:
                speaker_target = Segment(segment['start'], wav_duration)
            else:
                speaker_target = Segment(segment['start'], segment['end'])
            waveform_target, sample_rate = audio_pipe.crop(
                target_audio, speaker_target)
            embedding_target = embeddings_model(waveform_target[None])

            # compare embeddings using "cosine" distance
            distance = cdist(embedding_reference, embedding_target, metric="cosine")
            segment['cosine_dist'] = distance[0][0]

            # save back the info to the dict
            segment['is_candidate'] = True if distance[0][0] <= cdist_threshold else False
        logging.info(f'Finished speaker verification for: "{each.candidate_name}"')
        # ---- speaker verification ends ---- #

        # save result dictionary to local disk
        save_dict_to_disk(result, each, each.candidate_name,
                          output_dir, filename=each.id)

        # delete temp folders
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()