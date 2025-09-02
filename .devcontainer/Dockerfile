FROM mcr.microsoft.com/devcontainers/python:3.11

RUN apt-get update && apt-get install -y \
  build-essential \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyter ipywidgets numpy pandas matplotlib scikit-learn seaborn polars

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
