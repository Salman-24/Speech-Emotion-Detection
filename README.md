# Speech-Emotion-Detection

## Goal:
* To build an emotion detection system that could detect emotions from the speech.

## Dataset:
* IEMOCAP - Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset is used for training.
*	It contains approximately 12 hours of speech data i.e a total of 10,039 audio files which are of the format .opus. 
*	These audio files capture 11 different types of emotions. They are angry, happy, sad, neutral, frustrated, excited, surprised, fear, others, disgust and xxx.
*	Also, I created a sample of audio files for validation which captures a range of emotion.

## Approach implemented so far:
*	Firstly, I’ll be using a self-supervised pretrained model called Wav2Vec to obtain the features of speech data.
*	By using Wav2Vec, the raw speech or audio is converted into an embedding/representation that can be fed into a model. 
*	In order to get the embeddings of all the audio files, Wav2Vec expects the audio files to be in .wav format. So I used **Ffmpeg-Python** to convert all the audio files from .opus to .wav. Then I extracted the embeddings/representations of all the audio files using a base Wav2Vec model and stored the embeddings in my drive as **embeddings.zip**. The code for this is present [here](https://github.com/Salman-24/Speech-Emotion-Detection/blob/main/Saving_Embeddings.ipynb) and requirements to run this code is present [here](https://github.com/Salman-24/Speech-Emotion-Detection/blob/main/requirements1.txt)
*	Now that I have my embeddings, I can use them in order to train my simple SVM classifier. But the problem I faced here is that, since the audio files are of varying duration the dimension/size of each embedding is different. So training the SVM classifier with different embedding sizes is not possible. 
*Example:* The embedding size for two audio files is *(1,118,32)* and *(1,78,32)*. As you can see, the embedding size was different in one axis of dimension i.e *(1,118)* & *(1,78)* and was constant in the other axis *(1,32)* & *(1,32)*. So I had to work around a specific dimension of the vectors before training the SVM so that all the embeddings were of the same size i.e **(1,32)**. Now that I’ve converted the embedding size of all the audio files to *(1,32)* I was able to train my SVM classifier. But by doing this a lot of data was lost, so the initial accuracy observed was very less. The code for this is present [here](https://github.com/Salman-24/Speech-Emotion-Detection/blob/main/Support_Vector_Machine.ipynb)
* So in order to solve the above issue I found a [repository](https://github.com/WinsteadZhu/Fine-Tune-Wav2Vec2) on GitHub that uses the same Wav2Vec model along with training an MLP of 2 layers but on the RAVDESS dataset. So I just swapped datasets here as to how the *data.json* expects in the model and proceeded with my training. The IEMOCAP dataset is stored in the drive as **IEMOCAP.zip**. Code for this is present [here](https://github.com/Salman-24/Speech-Emotion-Detection/blob/main/Wav2vec2_0_IEMOCAP.ipynb) and requirements to run this code is present [here](https://github.com/Salman-24/Speech-Emotion-Detection/blob/main/requirements2.txt)
* Since the IEMOCAP dataset is large, as I was working on my training process, I would end up reaching the session limits for the free-tier of Colab GPU compute because I was tagged as a long-running computation user. So the sessions were interrupted frequently during the training. In order to overcome this I subscribed to the paid version of Colab called Colab Pro.
*	Lastly as a part of the training process I augmented the dataset by splitting up audio files longer than 7 seconds into smaller splits of overlapping emotions and also removed the silent zones to have denser representations. I did this using a tool called **audacity**. I also removed the audio files with emotions ***xxx*** because the files did not capture a particular emotion. Audio files with emotion ***fear*** and ***disgust*** were also removed because there were only 5 audio files with these emotions. The augmented dataset is stored in drive as **final_dataset.zip**

## Next Steps

* In order to improve the initial accuracy, hyper-parameter tuning & configuration of different optimizers should be done. Also, increasing the number of epochs & reloading the last checkpoint of the model will also help in improving the accuracy.








