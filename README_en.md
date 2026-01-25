# Python Workspace Template

[Japanese Version](README.md) | English

This repository is intended for algorithm development and backtesting for stock auto-trading.
Provides a modern development environment using Dev Container, uv, Ruff, and Mypy.

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
