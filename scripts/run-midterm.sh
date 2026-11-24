#!/usr/bin/env bash
# Usage: bash scripts/run-midterm.sh /path/to/robomini_sim [/path/to/results]
set -eo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sim="$(cd -- "${1:?Provide the path to the official robomini_sim clone}" && pwd)"
out="$(realpath -m -- "${2:-$root/midterm-evidence}")"
side="${ROBOMINI_SIDE:-blue}"
mkdir -p "$out"

source /opt/ros/jazzy/setup.bash
if [[ ! -f "$sim/ros2_ws/install/setup.bash" ]]; then
    bash "$sim/scripts/build.sh"
fi
source "$sim/ros2_ws/install/setup.bash"
(
    cd "$root/ros2_ws"
    colcon build --symlink-install --packages-select robomini_vision
)
source "$root/ros2_ws/install/setup.bash"
export ROBOMINI_SIM_ROOT="$sim"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-41}"

sim_pid=
vision_pid=
cleanup() {
    if [[ -n "$vision_pid" ]]; then kill -INT "$vision_pid" 2>/dev/null || true; fi
    if [[ -n "$sim_pid" ]]; then kill -INT "$sim_pid" 2>/dev/null || true; fi
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ros2 launch robomini_sim sim.launch.py side:="$side" control_mode:=auto ammo_mode:=unlimited \
    auto_advance:=true gui:=false partition:="midterm-$$" \
    > "$out/gazebo.log" 2>&1 &
sim_pid=$!

ros2 run robomini_vision vision --ros-args -p use_sim_time:=true -p side:="$side" -p enable_fire:=true \
    > "$out/vision.log" 2>&1 &
vision_pid=$!

echo "Recording official Gazebo camera and referee output in $out"
timeout --signal=INT --kill-after=15s "${ROBOMINI_TIMEOUT:-300s}" ros2 run robomini_vision record \
    --ros-args -p use_sim_time:=true -p "output_dir:=$out" -p "display:=${ROBOMINI_DISPLAY:-false}"
