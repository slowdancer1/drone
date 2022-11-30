#!/usr/bin/env bash
set -x
set -p

export SDL_VIDEODRIVER_VALUE=offscreen
export SDL_HINT_CUDA_DEVICE=0
export DISPLAY=:0

cleanup() {
  kill $(pgrep -P $SIM_PID)
  wait
}

pushd /mnt/ssd/AirSimNH/LinuxNoEditor/
bash ./AirSimNH.sh &
SIM_PID=$!
trap cleanup EXIT
popd

sleep 6
timeout 120 python test_airsim.py --resume $1
# echo 
# wait
echo ~/Documents/AirSim/$(ls ~/Documents/AirSim -t | head -n1)
