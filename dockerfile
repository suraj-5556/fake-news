FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/model.pkl /app/models/model.pkl

RUN pip install -r requirements.txt

RUN pip install .

RUN python -m spacy download en_core_web_sm

EXPOSE 5000

#local
CMD ["python", "app.py"]  

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]