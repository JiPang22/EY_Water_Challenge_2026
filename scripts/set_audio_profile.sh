#!/bin/bash
MAC_UNDERSCORE="bluez_card.68_52_10_5C_26_E7"

# 1. 고음질(A2DP) 프로필 강제 전환 시도
pactl set-card-profile $MAC_UNDERSCORE a2dp-sink-sbc-xq 2>/dev/null || \
pactl set-card-profile $MAC_UNDERSCORE a2dp-sink

# 2. 기본 출력 장치로 설정
pactl set-default-sink $(pactl list sinks short | grep $MAC_UNDERSCORE | cut -f2)

# 3. 볼륨 50% 설정 (귀 보호)
pactl set-sink-volume $(pactl list sinks short | grep $MAC_UNDERSCORE | cut -f2) 50%

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
