import pandas as pd
import os
import numpy as np
import librosa

GENRES_CODES = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
                'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}


def loadTracksData(file_path, dataset_type = "small"):
    df = pd.read_csv(file_path, index_col=0, header=[0, 1])
    keep_columns = [('set', 'split'),
                    ('set', 'subset'),
                    ('track', 'genre_top')]
    df = df[keep_columns]
    df = df[df[('set', 'subset')] == dataset_type]
    df['track_id'] = df.index
    return df


def getAudioPaths(audio_dir, track_id):
    track_id_str = "{:06d}".format(track_id)
    return os.path.join(audio_dir, track_id_str[:3], track_id_str + ".mp3")

def createSpectogram(audio_dir, track_id):
    filename = getAudioPaths(audio_dir, track_id)
    time_series, sampling_rate = librosa.load(filename)
    spectogram = librosa.feature.melspectrogram(y=time_series,
                                                sr=sampling_rate,
                                                n_fft=2048,
                                                hop_length=1024)
    spectogram = librosa.power_to_db(spectogram, ref=np.max)
    return spectogram.T

def createDataArrays(audio_dir, data_frame):
    genres = np.array([], dtype=int)
    spectrograms = np.empty((0, 640, 128))
    for idx, row in data_frame.iterrows():
        track_id = int(row["track_id"])
        genre = GENRES_CODES[str(row[("track", "genre_top")])]
        try:
            spectrogram = createSpectogram(audio_dir, track_id)
            spectrograms = np.append(spectrograms, [spectrogram[:640, :]], axis=0)
            genres = np.append(genres, genre)
        except:
            print("Could not process track: {:}".format(track_id))
    return spectrograms, genres

def splitTrainTestValidation(data_frame):
    df_train = data_frame[data_frame[('set', 'split')] == 'training']
    df_valid = data_frame[data_frame[('set', 'split')] == 'validation']
    df_test = data_frame[data_frame[('set', 'split')] == 'test']
    return df_train, df_valid, df_test

def saveProcessedData(audio_dir, dir_path, df, name):
    data, labels = createDataArrays(audio_dir, df)
    np.savez(os.path.join(dir_path, "{:}_data".format(name)), data, labels)
    return

def prepareData(data_path, dataset_type = "small"):
    tracks_df = loadTracksData(os.path.join(data_path, "fma_metadata", "tracks.csv"))
    audio_dir = os.path.join(data_path, "fma_{:}".format(dataset_type))
    df_train, df_valid, df_test = splitTrainTestValidation(tracks_df)
    processed_data_dir = "processed_data"
    if not os.path.exists(os.path.join(data_path, processed_data_dir)):
        os.mkdir(os.path.join(data_path, processed_data_dir))

    for df, name in zip([df_test, df_valid, df_train], ["test", "validation", "train"]):
        saveProcessedData(audio_dir, os.path.join(data_path, processed_data_dir), df, name)
    return
