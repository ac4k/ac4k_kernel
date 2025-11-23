# ac4k_kernel


## Installation

### Install from source

```
git clone git@github.com:ac4k/ac4k_kernel.git
cd ac4k_kernel
pip install -r requirements.txt

# Install in development mode (editable install)
pip install -e . --no-build-isolation
```

## Contribution Guidelines

### Pre-commit Hooks

To ensure code quality, style consistency, and commit integrity, we use pre-commit. Install the hooks before contributing:

```
# Install clang-format
pip install clang-format

# Navigate to the project root directory
cd ac4k_kernel

# Install pre-commit hooks (runs on every commit)
pre-commit install

# (Optional) Install pre-push hooks (runs additional checks before pushing to remote)
pre-commit install --hook-type pre-push
```

> Explanation: Hooks will automatically run code formatting (black, isort), linting (flake8, pylint), and syntax checks. Commits that fail validation will be blocked—fix the issues before re-committing.

### Contribution Workflow

1. Fork the repository
2. Create a feature branch (git checkout -b feature/your-feature-name)
3. Make your changes (follow the code style guidelines)
4. Run tests locally (see Testing)
5. Commit your changes (pre-commit hooks will run automatically)
6. Push to your forked repository
7. Create a Pull Request (PR) to the dev branch of the original repository

## Development Environment

### Recommended Docker Image

For a consistent and reproducible development environment, use the following Docker image (includes CUDA 12.8 + cuDNN + Ubuntu 22.04):

```
# Pull the recommended image
docker pull nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Run the container (mount project directory for local development)
docker run -it --gpus all -v $(pwd):/ac4k_kernel -w /ac4k_kernel nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
