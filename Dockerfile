FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable live logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies needed for numpy + timezone support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Start the bot
CMD ["python", "bitget_crypto_bot.py"]
