# gradio
FROM python:3.10 as gradio

# Set the working directory in the container

# Set the working directory in the container
WORKDIR /app
RUN pip install --no-cache-dir -U poetry

# Copy all files from the current directory to the working directory in the container
COPY config.yml poetry.lock pyproject.toml /app/

RUN poetry config virtualenvs.create false
RUN poetry install --with=dev

COPY meta_prompt /app/meta_prompt/
COPY app /app/app/
RUN poetry install --with=dev

# Expose the port (if necessary)
EXPOSE 7860

# Run the script when the container launches
CMD python app/gradio_meta_prompt.py

# streamlit
FROM python:3.10 as streamlit

# Set the working directory in the container
WORKDIR /app
RUN pip install --no-cache-dir -U poetry

# Copy all files from the current directory to the working directory in the container
COPY config.yml poetry.lock pyproject.toml /app/

RUN poetry config virtualenvs.create false
RUN poetry install --with=dev

COPY meta_prompt /app/meta_prompt/
COPY app /app/app/
COPY app.py /app/app.py
RUN poetry install --with=dev

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit script when the container launches
CMD ["streamlit", "run", "app.py"]