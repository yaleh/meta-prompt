# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to the working directory in the container
COPY . /app/

# Install the required dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV API_KEY=""
ENV PROXY=""
ENV OPENAI_API_BASE=""
ENV OTHER_ARGS="--advanced_mode"
ENV ADVANCED_MODE="true"
ENV SERVER_NAME="0.0.0.0"

# Expose the port (if necessary)
EXPOSE 7860

# Run the script when the container launches
CMD /usr/local/bin/python3 meta_prompt.py --api_key=${API_KEY} --proxy=${PROXY} --openai_api_base=${OPENAI_API_BASE} --server_name=${SERVER_NAME} ${OTHER_ARGS}