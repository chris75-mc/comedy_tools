FROM python:3.9

WORKDIR /app
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
COPY bot.py ./
COPY credentials.json /app/credentials.json
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# Expose port 5000
EXPOSE 3000
ENV PORT 3000
#launch the bot
CMD ["python", "bot.py"]