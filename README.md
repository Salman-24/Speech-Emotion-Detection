# Speech-Emotion-Detection
This repository contains work on Speech emotion recognition using the IEMOCAP dataset. This dataset has 10,039 audio files which are of .opus format. I have used two approaches as of now.
1) Training a simple SVM classifier.
2) Training an MLP of two layers.

The issue I faced while training the SVM classifier was that since all the audio files were of varying durations the embeddings were different in one axis of dimension. Example;The shape of the embeddings were multidimensional i.e (1,118,32) and shape of the embeddings for two audio files were (1,118,32) and (1,78,32). So I had to work around a specific dimension of the vectors before training the SVM so that all the embeddings were of a the same size eg (1,32).

So in order to solve the above issue I found a Wav2Vec2.0 model being used along with training an MLP of 2 layers but on the RAVDESS dataset. So I just swapped the dataset to IEMOCAP here.

# Structure
1) Labelled_dataset.csv - Labelled dataset that contains the audio file names and emotion.
2) Saving_Embeddings.ipynb - Contains code that converts the audio files to .wav format using Ffmpeg-python and also generating the Wav2vec embeddings and saving them in the drive as embeddings.zip.
3) Support_Vector_Machine.ipynb - Contains code that uses the wav2vec embeddings, averages them from all windows and then we concatenate the final embeddings of shape (1,32) with the labelled dataset to train a simple SVM Classifier. 
4) Wav2vec2_0_IEMOCAP.ipynb - Contains code that uses Wav2vec along with training an MLP of two layers. The dataset is uploaded in the drive as IEMOCAP.zip and also two packages data.py and model.py should be used before running this code. They are also uploaded in the drive.
5) Wav2vec2_0_RAVDESS.ipynb - Contains code that uses Wav2vec along with training an MLP of two layers. The dataset is uploaded in the drive as Data_CNN_ravdess_savee.zip and also two packages data.py and model.py should be used before running this code. They are also uploaded in the drive.
