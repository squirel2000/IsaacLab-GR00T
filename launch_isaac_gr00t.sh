#!/bin/bash

# Script to launch Isaac GR00T server and client in separate terminals
# Location: ~/Gits/IsaacLab-GR00T/launch_isaac_groot.sh

# Set the base directory
BASE_DIR="$HOME/Gits/IsaacLab-GR00T"
SERVER_DIR="$BASE_DIR/Isaac-GR00T"
CLIENT_DIR="$BASE_DIR/IsaacLab_ming"

# Function to check if server is already running
check_server_running() {
    # Check if the inference_service_g1.py process is running
    if pgrep -f "inference_service_g1.py --server" > /dev/null; then
        return 0  # Server is running
    else
        return 1  # Server is not running
    fi
}

# Function to launch server in new terminal
launch_server() {
    echo "Launching server in new terminal..."
    # Using gnome-terminal (default on Ubuntu)
    gnome-terminal --title="Isaac GR00T Server" --working-directory="$SERVER_DIR" -- bash -c "
        echo 'Activating isaaclab environment and starting server...'
        conda activate isaaclab
        python scripts/inference_service_g1.py --server
        echo 'Server stopped. Press Enter to close terminal.'
        read
    "
}

# Function to launch client in new terminal
launch_client() {
    echo "Launching client in new terminal..."
    # Wait a moment for server to initialize
    sleep 3
    gnome-terminal --title="Isaac GR00T Client" --working-directory="$CLIENT_DIR" -- bash -c "
        echo 'Activating isaaclab environment and starting client...'
        conda activate isaaclab
        ./isaaclab.sh -p scripts/gr00t_script/gr00t_infer_agent.py
        echo 'Client stopped. Press Enter to close terminal.'
        read
    "
}

# Main script execution
echo "Isaac GR00T Launcher"
echo "===================="

# Check if directories exist
if [ ! -d "$SERVER_DIR" ]; then
    echo "Error: Server directory not found: $SERVER_DIR"
    exit 1
fi

if [ ! -d "$CLIENT_DIR" ]; then
    echo "Error: Client directory not found: $CLIENT_DIR"
    exit 1
fi

# Check if server is already running
if check_server_running; then
    echo "Server is already running. Skipping server launch."
else
    echo "Server not found. Starting server..."
    launch_server
fi

# Launch client
echo "Starting client..."
launch_client

echo "Both terminals launched successfully!"
echo "Server terminal: Isaac GR00T Server"
echo "Client terminal: Isaac GR00T Client"