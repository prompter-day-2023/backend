FROM python:3.9.13

RUN mkdir /backend

WORKDIR /backend

COPY requirements.txt . /backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

COPY . .

EXPOSE 5123

# gunicorn -k gevent -w 4 app:app
CMD ["gunicorn", "-k", "gevent", "-w", "4", "--bind", "0.0.0.0:5123", "app:app"]