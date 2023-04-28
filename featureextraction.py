from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import librosa

class AudioFeatureExtractor:
    def __init__(self, csv_path, data_dir):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.head(8)
        self.feature_matrix = []
        
    def extract_features(self):
        for index, row in tqdm(self.df.iterrows(), total=8, desc="Extracting features"):
            audio_file = self.data_dir + "/" + row['filename']
            label = row['common_name']
            longitude = row['longitude']
            latitude = row['latitude']
            # Load the audio file and extract features
            y, sr = librosa.load(audio_file)
            # Data augmentation - pitch shifting
            pitch_shifted = librosa.effects.pitch_shift(y = y,sr =  float(sr), n_steps= 2.0)
            # Data augmentation - adding background noise
            noise = np.random.randn(len(y))
            noisy_audio = y + 0.1*noise
            # Data augmentation - time stretching
            stretched = librosa.effects.time_stretch(y, rate=0.8)
            features = librosa.feature.mfcc(y=y, sr=sr)
            # Append the features and label to the feature matrix
            self.feature_matrix.append((features, longitude, latitude, label))
            print(audio_file)
            print()
        
        # Convert the feature matrix to a numpy array
        feature_matrix = np.array(self.feature_matrix)
        
        # Extract the MFCC features
        mfcc_features = feature_matrix[:, 0]
        labels = feature_matrix[:, 3]
        geospatial_features = feature_matrix[:, 1:3]
        print(labels.shape)
        print(mfcc_features.shape)
        print(geospatial_features.shape)
        
        print("scaling MFCC features")
        # Normalize the MFCC features
        mfcc_scaler = StandardScaler()
        mfcc_features = mfcc_scaler.fit_transform(mfcc_features)
        
        # Extract the geospatial features
        print("scaling geospatial features")
        # Normalize the geospatial features
        scaler = StandardScaler()
        geospatial_features = scaler.fit_transform(geospatial_features)
        
        return mfcc_features, geospatial_features, labels
    
    

