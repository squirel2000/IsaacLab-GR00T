# IsaacLab + Isaac-GR00T + NoMachine Setup (Method 3)

This guide sets up:
- Ubuntu 22.04 LTS with NVIDIA RTX 4090
- Miniforge (conda)
- IsaacLab + Isaac-GR00T Python environment
- NoMachine using **virtual X session** (Method 3)

> ✅ Recommended for server setups without always-connected display (headless)

---
## GRUB Editing Method for NVIDIA RTX 4090 During Ubuntu Installation

### Step 1: Access GRUB Editor
- During system startup, when the purple "Try or Install Ubuntu" menu appears:
  - Press `Esc` or `Shift` to bring up the menu if needed.
  - Highlight the "Try or Install Ubuntu" option.
  - Press `e` to enter the GRUB editing mode.

### Step 2: Modify Kernel Parameters
- Locate the line starting with `linux` that contains `quiet splash`.
- **Add `nomodeset` directly after `quiet splash` without removing existing parameters**, for example:
`linux /boot/vmlinuz... quiet splash nomodeset $vt_handoff`


### Step 3: Boot with Modified Settings
- Press `Ctrl+X` or `F10` to boot and start the installation process.

---

### Notes for NVIDIA RTX 4090
- Adding `nomodeset` disables the default Nouveau driver, which does not support RTX 40-series cards.
- This allows the installer to run without graphical issues.
- After installation, you must install the proprietary NVIDIA drivers and remove `nomodeset` to enable full GPU functionality.

---

### Post-Installation Driver Setup (Summary)
sudo apt update
sudo apt install nvidia-driver-570
sudo nano /etc/default/grub # Remove "nomodeset" from GRUB_CMDLINE_LINUX_DEFAULT
sudo update-grub
sudo reboot


---

## 🔧 Step 1: Install Miniforge (conda)
```bash
cd ~/Downloads
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
rm Miniforge3-Linux-x86_64.sh
```

### Disable auto-activation of `base`:
```bash
conda config --set auto_activate_base false
```

### Initialize Conda in bash shell:
```bash
~/miniforge3/bin/conda init bash
```

---

## 🧪 Step 2: Create and Setup IsaacLab Conda Environment
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n isaaclab python=3.10
conda activate isaaclab
```

### Install core dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install isaacgymenvs
```

### Clone and install IsaacLab:
```bash
mkdir -p ~/Gits && cd ~/Gits
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git
cd IsaacLab
pip install -e .
```

---

## 🤖 Step 3: Install Isaac-GR00T
```bash
cd ~/Gits
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install -e .
```

---

## 🖥️ Step 4: Configure NoMachine (Virtual X Session Mode - Method 3)

### Stop display manager:
```bash
sudo systemctl stop display-manager
sudo /etc/NX/nxserver --restart
```

> This allows NoMachine to start its **own virtual X session**, independent of any physical monitor.

### Optional: Disable display manager permanently to always use virtual X
```bash
sudo systemctl disable display-manager
```

---

## 🔁 Step 5: Auto-Restart NoMachine Virtual Desktop on Boot

Create a systemd service:
```bash
sudo tee /etc/systemd/system/nomachine-vd.service > /dev/null <<EOF
[Unit]
Description=NoMachine Virtual Desktop auto-start
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/NX/bin/nxserver --restart
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nomachine-vd.service
```

---

## ✅ Final Notes
- After reboot, NoMachine will launch a virtual X session automatically.
- On first connection, select **"Create a new virtual desktop"**.
- Set resolution to `1920x1080` in NoMachine Display settings.
- Activate IsaacLab environment with:

```bash
conda activate isaaclab
```

> You can add `conda activate isaaclab` to `.bashrc` to make it default in all terminals.
