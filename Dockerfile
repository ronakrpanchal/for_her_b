FROM python:3.12-slim

# Install PDM
RUN pip install pdm

# Set working directory
WORKDIR /app

# Copy PDM configuration files first (for better Docker layer caching)
COPY pyproject.toml pdm.lock* ./

# Configure PDM to use system Python and install dependencies
RUN pdm config python.use_venv false
RUN pdm install --prod --no-editable

# Copy all application code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app


# Start the application using system Python instead of PDM's venv
CMD ["sh", "-c", "python -m uvicorn main:app"]