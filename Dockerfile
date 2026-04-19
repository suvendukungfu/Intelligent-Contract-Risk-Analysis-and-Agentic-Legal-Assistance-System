# Use a specific, updated parent image (Bookworm is more secure)
FROM python:3.11-slim-bookworm

# Set environment variables for better container behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies
# We include libsqlite3-dev for the pysqlite3 build
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (mitigates high vulnerabilities)
RUN groupadd -r streamlit && useradd -r -g streamlit -d /app streamlit \
    && chown -R streamlit:streamlit /app

# Copy and install Python dependencies
COPY --chown=streamlit:streamlit requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Download the Spacy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY --chown=streamlit:streamlit . .

# Switch to the non-root user
USER streamlit

# Expose the standard port (Hugging Face expects 7860)
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
