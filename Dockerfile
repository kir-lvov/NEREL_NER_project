FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

COPY src/ ./src/
COPY notebooks/ ./notebooks/

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]