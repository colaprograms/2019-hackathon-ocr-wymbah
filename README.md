### An OCR for small pictures with a restricted set of symbols

An OCR program for small handwritten numbers like 135.8 or 4,2. The pictures come from [AI4Good---Meza-OCR-Challenge](https://github.com/Charitable-Analytics-International/AI4Good---Meza-OCR-Challenge). We use CNN-RNN-CTC with Resnet-34 and a single bidirectional LSTM.

<s>Try out the notebooks!</s> **Note: These don't work anymore because the github repository containing the data set has been deleted, but this repository contains trained models, so you can still use beamtst.ipynb to OCR your own handwritten images from those models.**

* [training.ipynb](https://github.com/colaprograms/2019-hackathon-ocr-wymbah/blob/master/notebooks/training.ipynb) trains a model.
* [beamtst.ipynb](https://github.com/colaprograms/2019-hackathon-ocr-wymbah/blob/master/notebooks/beamtst.ipynb) shows you how to use it.

The repo comes with a couple of trained models already, as you can see in the second notebook `beamtst.ipynb`.

**NOTE**: This is not the code that we submitted in the hackathon. This is the codebase that I was building, which took an extra day to debug. The hackathon code is available from here: https://devpost.com/software/wymbah-helping-read-doctor-s-handwritting
