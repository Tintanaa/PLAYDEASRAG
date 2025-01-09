FROM python:3.10

# Create app directory
WORKDIR /app

# Copy all files except the excluded folder
ARG EXCLUDE_FOLDER
COPY . .

# Exclude specified folder
RUN echo "Excluding folder: ${EXCLUDE_FOLDER}" && rm -rf $EXCLUDE_FOLDER

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
