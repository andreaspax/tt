FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY vocab.json .
COPY id_vectors.npy .
COPY epoch2_query_tower.pt .
COPY epoch2_document_tower.pt .
COPY document_final_embeddings_val.pkl .
COPY unique_documents_val.pkl .

COPY *.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]