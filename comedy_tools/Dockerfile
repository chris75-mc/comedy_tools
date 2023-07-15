FROM python:3.9

WORKDIR /app
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
COPY src ./src
COPY data ./data
COPY flask_api.py ./
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# Expose port 5000
EXPOSE 3000
ENV PORT 3000
# Use gunicorn as the entrypoint
CMD exec gunicorn --bind :$PORT flask_api:app --workers 1 --threads 1 --timeout 60