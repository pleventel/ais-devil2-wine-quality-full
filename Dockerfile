ARG PYTHON_VERSION=3.13

# Builder stage: install dependencies
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Never let uv download its own Python, always use the one of the image
ENV UV_PYTHON_DOWNLOADS=never

# Install curl and uv
RUN apt-get update && apt-get install -y curl
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy only the files that define the dependencies (-> layer caching)
COPY pyproject.toml uv.lock ./

# Install exactly the locked versions into /app/.venv
RUN uv sync --frozen --no-dev

# Runtime stage: clean image with only what we need to run the API
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Take the installed environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Take only the code and the model
COPY wine_quality_api.py wine_quality_model.pkl ./

EXPOSE 8000

ENTRYPOINT ["/app/.venv/bin/uvicorn"]
CMD ["wine_quality_api:app", "--host", "0.0.0.0", "--port", "8000"]
