FROM pytorch/pytorch:latest

COPY . /usr/app/
WORKDIR /usr/app/
EXPOSE 8000

RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

CMD['python', 'api.py']

