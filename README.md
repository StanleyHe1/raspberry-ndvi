# Raspberry Pi NDVI Analyzer

This project captures images from two Raspberry Pi cameras — one **RGB** and one **NoIR (Near-Infrared)** — to compute the **Normalized Difference Vegetation Index (NDVI)**, a measure of plant health based on light reflectance.

---

## Requirements

### Hardware
- Raspberry Pi 4 (recommended) or Pi 3B+
- Two Raspberry Pi Camera Modules (one RGB, one NoIR)
- Dual-camera adapter (if using two CSI ports)
- MicroSD card (32 GB or larger)

### Software
- Raspberry Pi OS (Bookworm or Bullseye)
- Python 3.9 or newer

- Install system-level and Python dependencies:
    ```bash
    sudo apt update
    sudo apt install python3-opencv python3-picamera2 -y
    pip3 install -r requirements.txt
