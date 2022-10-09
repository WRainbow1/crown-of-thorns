FROM gcr.io/deeplearning-platform-release/tf-gpu.2-8
# FROM python:3.9.13


CMD nvidia-smi
#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip

ADD main.py .

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3","-u","./main.py"]