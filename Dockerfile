FROM frolvlad/alpine-python-machinelearning:latest
#Alpine Linux-based image with Python and machine learning libraries pre-installed

RUN pip install --upgrade pip

WORKDIR /app
RUN apk add --no-cache git
COPY . /app

RUN apk add build-base
RUN apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    apk add --no-cache --update python3 && \
    pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

EXPOSE 8000

ENTRYPOINT  ["python"]
CMD ["refregirator.py"]