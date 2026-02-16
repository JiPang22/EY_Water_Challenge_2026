#!/bin/bash
# 1. 로그에서 확인된 정확한 장치명으로 기본값 설정
SINK_NAME="bluez_output.68_52_10_5C_26_E7.1"

# 2. 기본 오디오 장치 강제 지정
pactl set-default-sink $SINK_NAME

# 3. 볼륨 40% 설정 (깜짝 놀람 방지)
pactl set-sink-volume $SINK_NAME 40%

# 4. 음소거 해제 (혹시 모를 Mute 방지)
pactl set-sink-mute $SINK_NAME 0

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
