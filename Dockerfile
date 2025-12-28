# FROM python:3.10-slim-bookworm
FROM trading_base:latest

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Switch to root to install packages if the base image uses a non-root user
USER root

# 1. System Dependencies: Xvfb, Java, FFmpeg
# We need libxrender1, libxtst6, libxi6 for Java AWT/Swing in headless mode (even with Xvfb)
RUN apt-get update && apt-get install -y \
    openjdk-17-jre \
    xvfb \
    ffmpeg \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python Environment: Ray RLLib, PyTorch, Gym, OpenCV, MSS
# First install everything except robocode-tank-royale
# Removed torch as it is in base image
RUN pip install --upgrade pip && \
    pip install \
    opencv-python-headless \
    mss \
    pyautogui \
    websockets \
    hydra-core


# Install robocode-tank-royale Python bot API from PyPI
RUN pip install --no-cache-dir robocode-tank-royale

# 3. Game Artifacts setup
WORKDIR /robocode
# Use v0.34.1 - server and GUI verified to exist  
RUN echo "Downloading v0.34.2 jars..." && \
    wget https://github.com/robocode-dev/tank-royale/releases/download/v0.34.2/robocode-tankroyale-server-0.34.2.jar -O server.jar && \
    wget https://github.com/robocode-dev/tank-royale/releases/download/v0.34.2/robocode-tankroyale-gui-0.34.2.jar -O gui.jar
# Note: booter.jar not available for v0.34.2, but we don't need it (running Python bots directly)

# 4. Environment Configuration
ENV DISPLAY=:99
ENV ROBOCODE_SERVER_URL=ws://localhost:7654
ENV JAVA_OPTS="-Xms256m -Xmx512m"

# 5. Copy Agent Code and Config
COPY src /app/src
COPY config /app/config
COPY sample_bots /app/sample_bots
COPY start_sim.sh /app/start_sim.sh
WORKDIR /app

# 6. Entrypoint Script
RUN chmod +x /app/start_sim.sh
# Default command if none provided
CMD ["/app/start_sim.sh"]
