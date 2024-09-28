# Swahili Automatic Speech Recognition (ASR) Model

## Overview

This repository contains code and instructions for utilizing a Swahili Automatic Speech Recognition (ASR) model based on the transformers library. The model is pre-trained on Swahili audio data and can be used to transcribe Swahili speech in various applications.

## Instructions

### 1. Install Dependencies

Make sure you have the required dependencies installed. You can install them by running the following command:

```bash
!pip install transformers
```

### 2. Set Up the ASR Pipeline

```python
from transformers import pipeline
from google.colab import drive

# Set up the ASR pipeline using the Swahili model
# model example is Akash/Swahili_xlrs from hugging face, I also used alamsher/wav2vec2-large-xlsr-53-common-voice-sw
pipe = pipeline("automatic-speech-recognition", model="Akashpb13/Swahili_xlsr", device=0)

# Mount Google Drive to access your files
drive.mount('/content/drive')
```

### 3. Load and Transcribe Audio Files

#### Single Audio File:

```python
from IPython.display import Audio

# Display an audio file
Audio("/content/test/common_voice_sw_27729935.mp3")

# Transcribe the audio file
pipe("/content/test/common_voice_sw_27729935.mp3")
```

#### Multiple Audio Files:

```python
# Create a DataFrame with file paths
test = pd.read_csv('/content/drive/MyDrive/Models/SampleSubmission.csv')
test["my_path"] = ["/content/test/" + i for i in test.path]

# Transcribe multiple audio files
pipe(test.my_path.to_list())
```

### 4. Save Transcriptions to CSV

```python
# Create a submission DataFrame
sub = pd.DataFrame()
sub["path"] = test.path.to_list()
sub["sentence"] = result_list

# Save the submission to a CSV file
sub.to_csv("ASR_Submission.csv", index=False)
```

### 5. Perform ASR on Entire Dataset

```python
# Perform ASR predictions on the entire dataset
res = []
for path in tqdm(test.path):
    res.append(pipe(f'/content/drive/MyDrive/asr/test_audios/{path}')['text'])

# Update the test DataFrame with the transcribed sentences
test['sentence'] = res

# Save the DataFrame to a CSV file for further analysis
test[['audio_ID', 'sentence']].to_csv('/content/drive/MyDrive/asr/res.csv', index=False)
```

### Note:

- Make sure to replace file paths and names according to your specific setup.
- The provided code assumes a Colab environment. If you are using a different environment, adjust the code accordingly.
- For further details on the transformers library, refer to the [official documentation](https://huggingface.co/transformers/).

---

Feel free to customize the README based on your specific needs and audience.
