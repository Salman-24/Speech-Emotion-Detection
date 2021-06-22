import os
import librosa
import statistics
import numpy as np
import pickle
# import pandas as pd
import seaborn as sns
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pydub.utils import make_chunks
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


# loading json and creating model
from tensorflow.keras.models import model_from_json
json_file = open('model5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Emotion_Voice_Detection_Model5.h5")
print("Loaded model from disk")


pkl_file = open('encoder.pkl', 'rb')
encoder = pickle.load(pkl_file)
pkl_file1 = open('scaler.pkl', 'rb')
scaler = pickle.load(pkl_file1)

myaudio = AudioSegment.from_file("Youtube_audios/un1.mp3","mp3") 
chunk_length_ms = 4000
chunks = make_chunks(myaudio, chunk_length_ms)
try:
  os.rmdir('chunks')
except:
  pass

if not os.path.exists('chunks'):
    os.makedirs('chunks')
for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}".format(i)
    wav_filename = './chunks/' + chunk_name + '.wav'
    chunk.export(wav_filename, format="wav")

pathAudio="./chunks"
files = librosa.util.find_files(pathAudio, ext=['wav']) 
# files = np.asarray(files)
# data_path= pd.DataFrame(files,columns=['wav_file'])

def extract_features(data):
    result = np.array([])
    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))
    mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
    result = np.hstack((result, mel))
    return result
def get_features(path):
    data, sample_rate = librosa.load(path)
    res1 = extract_features(data)
    result = np.array(res1)
    return result


X = []
for path in files:
  feature = get_features(path)
  X.append(feature)



# Cols = pd.DataFrame(X)
# x1 = Cols.iloc[: ,:].values
# x2 = scaler.transform(x1)
x2 = scaler.transform(X)
x3 = np.expand_dims(x2, axis=2)


def approach1(decodedarrays):
  print('Approach1')

  livepreds=loaded_model.predict(decodedarrays)
  livepreds1=livepreds.argmax(axis=1)
  liveabc = livepreds1.astype(int).flatten()
  b = (encoder.inverse_transform((liveabc)))

  sat = []
  notsat = []
  for i in range(0,len(b)):
    if b[i]=='exc' or b[i]=='hap' or b[i]=='neu':
      if i <= len(b)-3:
        sat.append([b[i],10])
      elif i == len(b)-1:
        sat.append([b[i],30])
      else:
        sat.append([b[i],20])
    else:
      if i <= len(b)-3:
        notsat.append([b[i],10])
      elif i == len(b)-1:
        notsat.append([b[i],30])
      else:
        notsat.append([b[i],20])

  satsum = 0
  for i in range(0,len(sat)):
    satsum = sat[i][1]+satsum
  # print(satsum)

  notsatsum = 0
  for i in range(0,len(notsat)):
    notsatsum = notsat[i][1]+notsatsum
  # print(notsatsum)

  if satsum > notsatsum:
    return "Customer was Satisfied"
  elif satsum == notsatsum:
    if b[-1]=='ang' or b[-1]=='fru' or b[-1]=='sad':
      return "Customer was not satisfied"
    else:
      return "Customer was Satisfied"
  else:
    return "Customer was not Satisfied"

def approach2(decodedarrays):
  print('Approach2')

  satisfied= []
  notsatisfied = []
  for i in loaded_model.predict(decodedarrays):
    if np.argmax(i)==0 or np.argmax(i)==2 or np.argmax(i)==5:
      notsatisfied.append(max(i))
    else:
      satisfied.append(max(i))
  if not satisfied:
    satisfiedmean = 0
  else:
    satisfiedmean=statistics.mean(satisfied)
  if not notsatisfied:
    notsatisfiedmean = 0
  else:
    notsatisfiedmean=statistics.mean(notsatisfied)
  if satisfiedmean > notsatisfiedmean:
    return "Customer was satisfied"
  else:
    return "Customer was not satisfied"


print(approach1(x3))
print(approach2(x3))

# #Approach1
# Cols = pd.DataFrame(X)
# x1 = Cols.iloc[: ,:].values
# x2 = scaler.transform(x1)
# x3 = np.expand_dims(x2, axis=2)
# livepreds=loaded_model.predict(x3)
# livepreds1=livepreds.argmax(axis=1)
# liveabc = livepreds1.astype(int).flatten()
# b = (encoder.inverse_transform((liveabc)))
# sat = []
# notsat = []
# for i in range(0,len(b)):
#   if b[i]=='exc' or b[i]=='hap' or b[i]=='neu':
#     if i <= len(b)-3:
#       sat.append([b[i],10])
#     elif i == len(b)-1:
#       sat.append([b[i],30])
#     else:
#       sat.append([b[i],20])
#   else:
#     if i <= len(b)-3:
#       notsat.append([b[i],10])
#     elif i == len(b)-1:
#       notsat.append([b[i],30])
#     else:
#       notsat.append([b[i],20])

# satsum = 0
# for i in range(0,len(sat)):
#   satsum = sat[i][1]+satsum
# print(satsum)

# notsatsum = 0
# for i in range(0,len(notsat)):
#   notsatsum = notsat[i][1]+notsatsum
# print(notsatsum)

# if satsum > notsatsum:
#   print("Customer was Satisfied")
# elif satsum == notsatsum:
#   if b[-1]=='ang' or b[-1]=='fru' or b[-1]=='sad':
#     print("Customer was not satisfied")
#   else:
#     print("Customer was Satisfied")
# else:
#   print("Customer was not Satisfied")


# data=pd.read_excel("Approach1.xlsx")
# data

# y_pred = data['Predicted']
# y_test = data['Actual']
# print("Approach 1")
# print(classification_report(y_test, y_pred))

# from sklearn.metrics import confusion_matrix

# labels = ['satisfied', 'unsatisfied']
# cm = confusion_matrix(y_test, y_pred, labels)
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Approach 1')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# #Approach2
# Cols = pd.DataFrame(X)
# x1 = Cols.iloc[: ,:].values
# x2 = scaler.transform(x1)
# x3 = np.expand_dims(x2, axis=2)
# satisfied= []
# notsatisfied = []
# for i in loaded_model.predict(x3):
#   if np.argmax(i)==0 or np.argmax(i)==2 or np.argmax(i)==5:
#     notsatisfied.append(max(i))
#   else:
#     satisfied.append(max(i))
# if not satisfied:
#   satisfiedmean = 0
# else:
#   satisfiedmean=statistics.mean(satisfied)
# if not notsatisfied:
#   notsatisfiedmean = 0
# else:
#   notsatisfiedmean=statistics.mean(notsatisfied)
# if satisfiedmean > notsatisfiedmean:
#   print("Customer was satisfied")
# else:
#   print("Customer was not satisfied")



# data1=pd.read_excel("Approach2.xlsx")
# data1

# y_pred = data1['Predicted']
# y_test = data1['Actual']
# print("Approach 2")
# print(classification_report(y_test, y_pred))

# from sklearn.metrics import confusion_matrix

# labels = ['satisfied', 'unsatisfied']
# cm = confusion_matrix(y_test, y_pred, labels)
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Approach 2')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
