#!/bin/bash

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_message "Starting startup script"

# Function to download with progress
download_with_progress() {
    local url="$1"
    local output="$2"
    log_message "Starting download of $output"
    wget -q --show-progress -O "$output" "$url" 2>&1 | \
    sed -u 's/.* \([0-9]\+%\)\ \+\([0-9.]\+.\) \(.*\)/\1\n# Downloading... \1 (\2) \3/' | \
    while read line; do
        if [[ $line =~ ^[0-9]+% ]]; then
            percent=${line%\%}
            if (( percent % 25 == 0 )); then
                log_message "Download progress: $line"
            fi
        fi
    done
    log_message "Finished downloading $output"
}

# Set up environment
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Download models if they're not already present
MODELS_DIR="/app/models"
mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/flux1-schnell.safetensors" ]; then
    download_with_progress "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "$MODELS_DIR/flux1-schnell.safetensors"
fi

if [ ! -f "$MODELS_DIR/ae.safetensors" ]; then
    download_with_progress "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" "$MODELS_DIR/ae.safetensors"
fi

log_message "All models downloaded successfully"

# Start the Streamlit app
log_message "Starting Streamlit app"
streamlit run app.py --server.port=8080 --server.address=0.0.0.0