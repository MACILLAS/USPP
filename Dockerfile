FROM tensorflow/tensorflow:2.4.2-gpu

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \

RUN git git clone https://github.com/MACILLAS/USPP.git

WORKDIR /usr/src/app/USPP

COPY . .

CMD ["python", "./run.py"]
