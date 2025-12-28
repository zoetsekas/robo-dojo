#!/bin/bash
# Start Robocode infrastructure for data collection
# This starts: Xvfb, Server, GUI, 2 sample bots, and controller

set -e

echo "Starting Robocode infrastructure for data collection..."

# 1. Start Xvfb
export DISPLAY=:100
Xvfb :100 -screen 0 1024x768x24 &
XVFB_PID=$!
echo "Xvfb started on :100 (PID: $XVFB_PID)"
sleep 3

# 2. Start Robocode server
java -Djava.net.preferIPv4Stack=true -jar /robocode/server.jar --port 7654 &
SERVER_PID=$!
echo "Server started on port 7654 (PID: $SERVER_PID)"
sleep 10

# 3. Start GUI
java -Djava.net.preferIPv4Stack=true -Djava.awt.headless=false -jar /robocode/gui.jar \
  --server-url ws://127.0.0.1:7654 --no-sound &
GUI_PID=$!
echo "GUI started (PID: $GUI_PID)"
# 4. Start 2 minimal bots (observer will collect the data)
echo "Starting minimal NoOp bots..."

python /app/src/bots/noop_bot.py "Bot1" &
BOT1_PID=$!
echo "NoOp Bot1 started (PID: $BOT1_PID)"
sleep 3

python /app/src/bots/noop_bot.py "Bot2" &
BOT2_PID=$!
echo "NoOp Bot2 started (PID: $BOT2_PID)"
sleep 3

# 5. Start controller to trigger match (wait for 2 bots)
python -m src.env.robocode_controller ws://127.0.0.1:7654 &
CONTROLLER_PID=$!
echo "Controller started (PID: $CONTROLLER_PID)"
sleep 3

echo ""
echo "Infrastructure ready! Bots are battling..."
echo "Press Ctrl+C to stop"
echo ""

# Keep script running
wait
