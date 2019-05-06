FROM python:3.6

# Install OpenJDK-8
RUN \
    apt-get update  && \
    apt-get -y install openjdk-8-jdk && \
    pip3 install flask && \
    pip3 install nltk && \
    pip3 install numpy && \
    pip3 install sklearn && \
    pip3 install scipy

RUN pip install gdown
RUN apt-get install unzip

ARG GDRIVE_DL_LINK

RUN gdown https://drive.google.com/uc?id=${GDRIVE_DL_LINK}
RUN unzip -d . stanford-postagger-full-2016-10-31.zip

RUN [ "python", "-c", "import nltk; nltk.download('all')" ]

# Add local files and folders
ADD * /

EXPOSE 9651

CMD [ "python", "./starter.py" ]