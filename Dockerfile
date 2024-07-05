# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to the working directory in the container
COPY config.yml poetry.lock pyproject.toml /app/
COPY app /app/app/
COPY meta_prompt /app/meta_prompt/

RUN pip install --no-cache-dir -U poetry
RUN poetry config virtualenvs.create false
RUN poetry install --with=dev

# Expose the port (if necessary)
EXPOSE 7860

# Run the script when the container launches
CMD python app/gradio_meta_prompt.py