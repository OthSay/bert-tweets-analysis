FROM python:3.7

WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:./src"

ENV CONFIG_PATH "./config.json"

EXPOSE 8000

CMD ["python", "./src/app/app.py"]