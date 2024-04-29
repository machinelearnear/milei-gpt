import shutil
import json
import os
import pandas as pd
import logging
import argparse
import yt_dlp
from pathlib import Path
from tqdm import tqdm

# ---- logging ---- #
# set up root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
log_file = './logs/retrieve_data_youtube.log'
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

# ---- config ---- #
mapping = {
    # "milei presidente": "milei",
    "el peluca milei": "milei",
    # "javier milei": "milei",
    # "sergio massa": "massa",
    # "patricia bullrich": "bullrich",
    # "myriam bregman": "bregman",
    # "juan schiaretti": "schiaretti",
}

# ---- helper funcs ---- #
def extract_youtube_info(URL, search_these_names, save_dir):
    """
    Extracts YouTube information for a list of names and saves them to JSON files.

    Parameters:
    - URL (str): The URL format string to use for searching.
    - search_these_names (list): A list of names to search for.
    - save_dir (str): The directory to save the extracted JSON files.
    """
    def save_to_json(data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f)

    for name in tqdm(search_these_names):
        if "@" not in name:
            name = name.replace(' ','+').lower() 
        fname = f"{save_dir}/{name}.json"

        ydl_opts = {
            "extract_flat": True,
            "quiet": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                URL.format(name), download=False)
            save_to_json(info, fname)


class YouTubeDataProcessor:
    def __init__(self, directory):
        self.directory = directory

    def _json_to_df(self, fname):
        with open(fname, 'r') as file:
            data1 = [json.loads(line) for line in file if line.strip()]  # ensure line is not empty

        # exploding the 'entries' column
        df_exploded = pd.DataFrame(data1).explode('entries')

        # rename & delete columns
        df_exploded.drop(["id", "view_count"], axis='columns', inplace=True)
        df_exploded.rename(columns={
            # "channel": "channel_name",
            # "uploader_url": "channel_uploader_url",
            "title": "channel_title",
        }, inplace=True)

        # extracting the dictionaries in the 'entries' column into separate columns
        entries_df = df_exploded['entries'].apply(pd.Series)

        # concatenating the original columns with the new columns from 'entries' & deleting duplicate cols
        df = pd.concat([df_exploded.drop('entries', axis=1), entries_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # filter columns
        cols = [
            "channel_id",
            "channel",
            "uploader_url",
            "id",
            "url",
            "title",
            "duration",
            "view_count",
        ]

        return df[cols]

    def process_all_json_files(self):
        # list all JSON files in the directory
        all_files = [os.path.join(self.directory, fname) for fname in os.listdir(self.directory) if fname.endswith('.json')]

        # convert each JSON file to a dataframe and store in a list
        all_dfs = [self._json_to_df(fname) for fname in all_files]

        # concatenate all dataframes into a single dataframe
        final_df = pd.concat(all_dfs, ignore_index=True)

        # add names
        final_df['candidate_name'] = final_df['channel'].str.lower().map(mapping)

        return final_df


class YouTubeDataProcessor_Search:
    def __init__(self, directory):
        self.directory = directory

    def _json_to_df(self, fname):
        with open(fname, 'r') as file:
            data1 = json.load(file)

        # exploding the 'entries' column
        df_exploded = pd.DataFrame([data1]).explode('entries')

        # keep only the desired columns and rename the 'title' column to 'search_term'
        df_filtered = df_exploded[['title', 'extractor_key', 'entries']].rename(columns={'title': 'search_term'})

        # extracting the dictionaries in the 'entries' column into separate columns
        entries_df = df_filtered['entries'].apply(pd.Series)

        # concatenating the original columns with the new columns from 'entries' & deleting duplicate cols
        df = pd.concat([df_filtered.drop('entries', axis=1), entries_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        cols = [
            "search_term",
            # "extractor_key",
            "channel_id",
            "channel",
            "uploader_url",
            "id",
            "url",
            "title",
            "duration",
            "view_count",
        ]

        return df[cols]

    def process_all_json_files(self):
        # list all JSON files in the directory
        all_files = [os.path.join(self.directory, fname) for fname in os.listdir(self.directory) if fname.endswith('.json')]

        # convert each JSON file to a dataframe and store in a list
        all_dfs = [self._json_to_df(fname) for fname in all_files]

        # concatenate all dataframes into a single dataframe
        final_df = pd.concat(all_dfs, ignore_index=True)

        # add names
        final_df['candidate_name'] = final_df['search_term'].str.replace('"',"").map(mapping)

        return final_df


def main():
    parser = argparse.ArgumentParser(description='process and transcribe YouTube videos.')
    parser.add_argument('--data_dir_hq', default='../data/youtube_data_hq',
                       help='Directory containing JSON (default: ../data/youtube_hq)')
    parser.add_argument('--data_dir_lq', default='../data/youtube_data_lq',
                       help='Directory containing JSON (default: ../data/youtube_lq)')
    parser.add_argument('--output_filename', default='../data/dataset.csv',
                       help='File containing youtube video data (default: ../data/dataset.csv)')
    parser.add_argument('--min_duration', default=60*15, help='Min duration (default: 15 min)')
    parser.add_argument('--max_duration', default=60*150, help='Max duration (default: 150 min)')
    parser.add_argument('--min_view_count', default=100, help='Min video views (default: 100)')
    args = parser.parse_args()

    # assigning variables from argparse
    data_dir_hq = args.data_dir_hq
    data_dir_lq = args.data_dir_lq
    output_filename = args.output_filename
    min_duration = args.min_duration
    max_duration = args.max_duration
    min_view_count = args.min_view_count

    # ---- extract hq videos where candidate is the main speaker ---- #
    target_channels = [
        "@ElPelucaMilei",
        "@MILEIPRESIDENTE",
        "@JavierMileiOK",
        "@PatriciaBullrich",
        "@SergioMassa",
    ]

    save_dir_hq = Path(data_dir_hq)
    if save_dir_hq.exists(): shutil.rmtree(save_dir_hq)
    Path(save_dir_hq).mkdir(parents=True, exist_ok=True)
    URL = "https://www.youtube.com/{}/videos"
    extract_youtube_info(URL, target_channels, save_dir_hq)
    hq_processor = YouTubeDataProcessor(save_dir_hq)
    df_hq = hq_processor.process_all_json_files()
    df_hq["quality"] = "high"
    logging.info("finished processing `df_hq`")

    # ---- extract lq videos where candidate is the main speaker ---- #
    save_dir_lq = Path(data_dir_lq)
    if save_dir_lq.exists(): shutil.rmtree(save_dir_lq)
    Path(save_dir_lq).mkdir(parents=True, exist_ok=True)

    search_these_names = ["Javier Milei",
                          "Sergio Massa",
                          "Patricia Bullrich",
                          "Myriam Bregman",
                          "Juan Schiaretti"]

    URL = 'https://www.youtube.com/results?search_query=%22{}%22&sp=EgQQARgC'
    extract_youtube_info(URL, search_these_names, save_dir_lq)
    lq_processor = YouTubeDataProcessor_Search(save_dir_lq)
    df_lq = lq_processor.process_all_json_files()
    df_lq["quality"] = "low"
    logging.info("finished processing `df_lq`")

    # ---- merge both files ---- #
    # identify common columns
    common_columns = df_hq.columns.intersection(df_lq.columns).to_list()

    # subset dataframes using the common columns and append
    merged_df = pd.concat([df_hq[common_columns], df_lq[common_columns]], ignore_index=True)

    # ---- filter based on duration & delete duplicates ---- #
    subset = merged_df[(merged_df.duration > min_duration) & 
                   (merged_df.duration < max_duration) & 
                   (merged_df.view_count > min_view_count)].drop_duplicates(subset=['duration'])
    logging.info(subset.candidate_name.value_counts())

    # ---- export to disk ---- #
    subset.to_csv(output_filename, index=False)
    logging.info("saved file to `{}`".format(output_filename))


if __name__ == "__main__":
    main()