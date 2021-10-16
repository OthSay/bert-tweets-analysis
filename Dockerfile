FROM python:3.7

WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:./src"

EXPOSE 8000

CMD ["streamlit", "run", "src/app.py"]