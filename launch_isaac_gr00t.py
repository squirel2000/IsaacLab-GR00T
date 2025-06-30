#!/usr/bin/env python3
"""
Script to launch Isaac GR00T server and client in separate terminals
Location: ~/Gits/IsaacLab-GR00T/launch_isaac_groot.py
"""

import os
import argparse
import subprocess
import time
import psutil
from pathlib import Path

# Default arguments (matching the Python scripts' defaults)
MODEL_PATH="output/G1_CubeStacking_Dataset_Checkpoints_fft_bs1/"
TASK="Isaac-Stack-Cube-G1-Abs-v0"
SAVE_IMG_FLAG=False

class IsaacGrootLauncher:
    def __init__(self, cli_args):
        self.args = cli_args
        self.base_dir = Path.home() / "Gits" / "IsaacLab-GR00T"
        self.server_dir = self.base_dir / "Isaac-GR00T"
        self.client_dir = self.base_dir / "IsaacLab_ming"
        
    def check_server_running(self):
        """Check if the inference server is already running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('inference_service_g1.py' in cmd and '--server' in cmdline for cmd in cmdline):
                    print(f"Found running server process: PID {proc.info['pid']}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def launch_server(self):
        """Launch server in new terminal"""
        print("Launching server in new terminal...")

        # Construct server command with arguments
        server_script = "scripts/inference_service_g1.py"
        # Only pass model_path; other arguments will use their defaults from inference_service_g1.py
        server_args = f"--server --model_path '{self.args.model_path}'"

        # Launch with gnome-terminal
        subprocess.Popen([
            'gnome-terminal',
            '--title=Isaac GR00T Server',
            f'--working-directory={self.server_dir}',
            '--',
            'bash', '-c',
            f'echo "Activating isaaclab environment and starting server..."; ' \
            f'conda activate isaaclab; ' \
            f'python {server_script} {server_args}; ' \
            f'echo "Server stopped. Press Enter to close terminal."; read'
        ])
    
    def launch_client(self):
        """Launch client in new terminal"""
        print("Launching client in new terminal...")

        # Construct client command with arguments
        client_script = "scripts/gr00t_script/gr00t_infer_agent.py"
        # Only pass task; other arguments will use their defaults from gr00t_infer_agent.py
        client_args_list = [f"--task '{self.args.task}'"]
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
            f'echo "Activating isaaclab environment and starting client..."; ' \
            f'conda activate isaaclab; ' \
            f'./isaaclab.sh -p {client_script} {client_args}; ' \
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
    # inference_service_g1.py default model_path
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