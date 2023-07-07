FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.10-cuda11.3.1

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated \
    nano htop sudo build-essential \
    zip unzip wget curl

RUN pip install --upgrade pip
RUN pip install --upgrade nbformat
RUN pip install numpy setuptools \
    scipy scikit-learn pandas matplotlib opencv-python

RUN pip install midi-ddsp

RUN apt install ffmpeg libsndfile1-dev -y 
RUN pip install numpy==1.23
RUN pip install tensorflow==2.12.0
RUN pip install keras==2.12.0

RUN midi_ddsp_download_model_weights

CMD ["tail", "-f","/dev/null"]

## 