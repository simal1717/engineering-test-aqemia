FROM python:3.9.10
COPY . /app
EXPOSE 5000
WORKDIR /app 
RUN pip install -r ./requirements.txt
CMD ["python", "api.py"]