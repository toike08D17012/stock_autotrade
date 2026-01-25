# Python Workspace Template

[Japanese Version](README.md) | English

A template repository for Python project development.
Provides a modern development environment using Dev Container, uv, Ruff, and Mypy.

## Initial Setup
Please provide the following instructions to the Coding Agent. Note: Replace [XX] with the appropriate values for your project.

Plaintext
This repository is intended for [XX]. 
Please update the following items accordingly:

- Delete the "First Steps" / "Inital Setup" sections from both `README.md` and `README_en.md`.
- Update the repository description in `README.md` and `README_en.md`.
- In `docker/docker-compose.yml`, update the following to match the repository name:
    - Image name
    - Service name
    - volumes
    - working_dir
- In `docker/run-docker.sh`, update the service name to be launched to match the repository name.
- In `.devcontainer/devcontainer.json`, update the following to match the repository name:
    - name
    - service
    - workspaceFolder

## Features

- **Package Management**: High-speed dependency resolution using `uv`
- **Development Environment**: Consistent environment via Dev Container (`.devcontainer`)
- **Static Analysis / Formatting**: High-speed Lint/Format using `ruff`
- **Type Checking**: Static type checking using `mypy`
- **Machine Learning Support**: Pre-configured dynamic installation for PyTorch (CPU/CUDA)

## Usage

### 1. Start Dev Container
Open this repository in VS Code and start the container using the recommended "Dev Containers" extension.
`postCreateCommand` will automatically execute `uv sync` to set up the environment.

### 2. Add Dependencies
Use `uv` to add packages.

```bash
uv add <package_name>
```

### 3. Quality Control Commands
Based on `GEMINI.md`, we recommend the following command for quality checks in this project.

```bash
# Execute formatting, auto-fix linting, and type checking in one go
ruff format && ruff check --fix && mypy .
```

## Directory Structure

- `.devcontainer/`: Dev Container settings (for VS Code)
- `environments/python/`: Python project definition (`pyproject.toml` is placed here)
- `GEMINI.md`: Coding guidelines (Google Style, Ruff settings, etc.)
