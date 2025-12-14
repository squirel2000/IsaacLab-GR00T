#!/usr/bin/env python3
"""
Script to launch Isaac GR00T server and client in separate terminals
"""

import argparse
import subprocess
import psutil
from pathlib import Path

# Set the base directory and script paths
# Server
SERVER_SCRIPT="inference_service.py"
SERVER_CONDA_ENV="env_gr00t"
# Client
CLIENT_SCRIPT="gr00t_infer_agent.py"
CLIENT_CONDA_ENV="env_isaaclab"

# Default arguments (matching the Python scripts' defaults)
MODEL_PATH="output/G1_Inspire_Cabinet_Pour_Dataset_Checkpoints_N1_5_fft/"
TASK="Isaac-Cabinet-Pour-G1-Abs-v0"
SAVE_IMG_FLAG=False

class IsaacGrootLauncher:
    def __init__(self, cli_args):
        self.args = cli_args
        self.base_dir = Path.home() / "Gits" / "IsaacLab-GR00T"
        self.server_dir = self.base_dir / "Isaac-GR00T"
        self.client_dir = self.base_dir / "IsaacLab"
        
    def check_server_running(self):
        """Check if the inference server is already running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('inference_service.py' in cmd and '--server' in cmdline for cmd in cmdline):
                    print(f"Found running server process: PID {proc.info['pid']}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def launch_server(self):
        """Launch server in new terminal"""
        print("Launching server in new terminal...")

        # Construct server command with arguments
        server_script = f"scripts/{SERVER_SCRIPT}"
        # Only pass model_path; other arguments will use their defaults from inference_service.py
        server_args = f"--server --model_path '{self.args.model_path}' --embodiment_tag new_embodiment --data_config g1_can_pick_and_sort --denoising_steps 4"

        # Launch with gnome-terminal
        subprocess.Popen([
            'gnome-terminal',
            '--title=Isaac GR00T Server',
            f'--working-directory={self.server_dir}',
            '--',
            'bash', '-c',
            f'echo "Activating conda environment: {SERVER_CONDA_ENV}..."; ' \
            f'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {SERVER_CONDA_ENV} && python3 -u {server_script} {server_args}; ' \
            f'echo "Server stopped. Press Enter to close terminal."; read'
        ])
    
    def launch_client(self):
        """Launch client in new terminal"""
        print("Launching client in new terminal...")

        # Construct client command with arguments
        client_script = f"scripts/gr00t_script/{CLIENT_SCRIPT}"
        # Only pass task; other arguments will use their defaults from gr00t_infer_agent.py
        client_args_list = [f"--task '{self.args.task}' --filter"]
        if self.args.save_img:
            client_args_list.append("--save-img")

        client_args = " ".join(client_args_list)

        # Launch with gnome-terminal
        subprocess.Popen([
            'gnome-terminal',
            '--title=Isaac GR00T Client',
            f'--working-directory={self.client_dir}',
            '--',
            'bash', '-c',
            f'echo "Activating conda environment: {CLIENT_CONDA_ENV}..."; ' \
            f'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {CLIENT_CONDA_ENV} && python3 -u {client_script} {client_args}; ' \
            f'echo "Client stopped. Press Enter to close terminal."; read'
        ])
    
    def validate_directories(self):
        """Check if required directories exist"""
        if not self.server_dir.exists():
            print(f"Error: Server directory not found: {self.server_dir}")
            return False
        
        if not self.client_dir.exists():
            print(f"Error: Client directory not found: {self.client_dir}")
            return False
        
        return True
    
    def run(self):
        """Main execution function"""
        print("Isaac GR00T Launcher")
        print("====================")
        
        # Validate directories
        if not self.validate_directories():
            return False
        
        # Check if server is already running
        if self.check_server_running():
            print("Server is already running. Skipping server launch.")
        else:
            print("Server not found. Starting server...")
            self.launch_server()
        
        # Launch client
        print("Starting client...")
        self.launch_client()
        
        print("Both terminals launched successfully!")
        print("Server terminal: Isaac GR00T Server")
        print("Client terminal: Isaac GR00T Client")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Isaac GR00T server and client with configurable arguments.")

    # Arguments for server and client, with defaults matching their original scripts
    # inference_service.py default model_path
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the model checkpoint directory for the server.")
    # gr00t_infer_agent.py default task
    parser.add_argument("--task", type=str, default=TASK,
                        help="Name of the task for the client.")
    # gr00t_infer_agent.py save_img flag
    parser.add_argument("--save-img", action="store_true", default=SAVE_IMG_FLAG,
                        help="[Client] Save RGB camera images from the simulation.")

    cli_args = parser.parse_args()

    launcher = IsaacGrootLauncher(cli_args)
    launcher.run()
