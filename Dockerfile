FROM python:3-slim-buster

WORKDIR /ROCHE-DEMO-OPENAI

COPY requirements.txt .

COPY . .

RUN pip install -r requirements.txt

#RUN pip install matplotlib

#RUN pip install plotly

#RUN pip install python-dotenv

#RUN pip install tiktoken

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
#CMD [ "python", "main.py" ]
