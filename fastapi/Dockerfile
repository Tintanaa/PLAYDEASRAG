# Use official Python image as a base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create and activate venv
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Install ngrok
RUN apt-get update
RUN apt-get install -y wget unzip 
RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz 
RUN ls
RUN tar -xvzf ./ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin

# Copy the application code
COPY . .

# Load the .env file and set the NGROK_AUTH_TOKEN environment variable
RUN if [ -f .env ]; then export $(cat .env | xargs) && echo "NGROK_AUTH_TOKEN is set to: $NGROK_AUTH_TOKEN"; fi

# set auth token
RUN ngrok config add-authtoken ${NGROK_AUTH_TOKEN}

# Expose the FastAPI app's default port
EXPOSE 8000

# Command to run the FastAPI app with multiple workers
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
