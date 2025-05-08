FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir streamlit sympy scikit-learn joblib
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
