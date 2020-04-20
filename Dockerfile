FROM "ubuntu:bionic"

RUN apt-get update && yes | apt-get upgrade

RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0

RUN apt-get install -y git python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install opencv-python

RUN pip3 install tensorflow==1.15.0

RUN pip3 install fastai==1.0.52

RUN apt-get install -y protobuf-compiler python3-pil python3-lxml

RUN pip3 install matplotlib

RUN mkdir -p /tensorflow

RUN pip3 install aiofiles==0.4.0

RUN pip3 install uvicorn==0.7.1

RUN pip3 install aiohttp==3.5.4

RUN pip3 install asyncio==3.4.3

RUN pip3 install pillow~=6.0

RUN pip3 install python-multipart==0.0.5

RUN pip3 install starlette==0.12.0

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

COPY required_files /tensorflow/models/research/object_detection/required_files

WORKDIR /tensorflow/models/research

EXPOSE 5555

CMD ["python3", "/tensorflow/models/research/object_detection/required_files/app/server.py", "serve"]
