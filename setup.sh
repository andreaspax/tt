#!/bin/bash

# Wrapper function to catch errors and return from the function instead of exiting the terminal
wrapper() {
  # Set the specific Python version you want to use
  PYTHON_VERSION="3.11.11"

  # Catch errors and return from the function instead of exiting the terminal
  trap 'echo "Setup failed!"; return' ERR

  # Function to install pyenv if not already installed
  install_pyenv() {
    if ! command -v pyenv &> /dev/null; then
      echo "pyenv not found, installing pyenv..."
      if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew update
        brew install pyenv
      elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu
        # Install python build dependencies
        sudo apt update -y
        sudo apt install -y build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev curl git \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        # Install pyenv
        curl https://pyenv.run | bash
        # You may need to add pyenv to your shell configuration
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        source ~/.bashrc
      fi
    else
      echo "pyenv is already installed"
    fi
  }

  # Ensure pyenv is installed
  install_pyenv

  # Initialize pyenv (only after it's confirmed to be installed)
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"

  # Install the specified Python version
  echo "Installing Python $PYTHON_VERSION with pyenv..."
  pyenv install $PYTHON_VERSION --skip-existing

  # Set the local Python version for the project
  echo "Setting local Python version to $PYTHON_VERSION for this project"
  pyenv local $PYTHON_VERSION

  # Check if pyenv is using the correct version
  python_version=$(python --version)
  echo "Using Python version: $python_version"

  # Check for or create virtual environment
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
  else
    echo "Virtual environment already exists."
  fi

  # Activate the virtual environment
  echo "Activating virtual environment..."
  source venv/bin/activate

  # Install dependencies from requirements.txt
  if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
  else
    echo "requirements.txt file not found. Skipping pip install."
  fi

  echo "Setup complete!"
}

wrapper