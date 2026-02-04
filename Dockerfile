FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/

EXPOSE 8501

CMD sh -c 'streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0'
