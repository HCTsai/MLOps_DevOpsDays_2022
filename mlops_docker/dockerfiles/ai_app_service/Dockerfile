FROM python:3.8-slim-buster

WORKDIR /usr/src/app/web
COPY . ../
RUN pip install --no-cache-dir -r ../requirements.txt
CMD ["python", "./app.py" ]
EXPOSE 5000