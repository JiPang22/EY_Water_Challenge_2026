#!/bin/bash
# 1. Pipewire 소켓 및 서비스 강제 완전 정지
systemctl --user stop pipewire.socket pipewire-pulse.socket wireplumber.service
systemctl --user stop pipewire.service pipewire-pulse.service

# 2. 블루투스 장치 강제 연결 (장치 주소 기반)
MAC_ADDR="68:52:10:5C:26:E7"
bluetoothctl disconnect $MAC_ADDR
sleep 2
bluetoothctl connect $MAC_ADDR

# 3. 서비스 순차적 재시작
systemctl --user start pipewire.service pipewire-pulse.service wireplumber.service

# 4. A2DP(고음질) 프로필 강제 설정 시도
sleep 3
pactl set-card-profile bluez_card.68_52_10_5C_26_E7 a2dp-sink-sbc-xq 2>/dev/null || pactl set-card-profile bluez_card.68_52_10_5C_26_E7 a2dp-sink

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
