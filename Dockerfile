# Use a lightweight Python image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into the system python inside the container
RUN uv sync --frozen --system

# Copy the rest of your code
COPY . .

# We don't set a CMD here because we have 4 different entry points