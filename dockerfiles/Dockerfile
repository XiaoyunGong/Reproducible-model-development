FROM python:3.9-slim-buster

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y \
curl

COPY . /app

ENTRYPOINT ["python3"]