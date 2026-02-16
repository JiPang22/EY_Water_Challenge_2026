#!/bin/bash
# 1. WirePlumber 캐시 및 상태 완전 초기화 (장치 인식 꼬임 해결)
rm -rf ~/.local/state/wireplumber/*

# 2. 블루투스 오디오 프로필 모듈 강제 확인
sudo apt-get install -y libspa-0.2-bluetooth 2>/dev/null

# 3. 서비스 순차적 재시작 (소켓 포함)
systemctl --user stop wireplumber pipewire pipewire-pulse
systemctl --user start pipewire pipewire-pulse wireplumber

# 4. 장치 검색 대기 및 A2DP 프로필 강제 할당
echo "장치 재스캔 중... 5초만 기다려"
sleep 5

CARD=$(pactl list cards short | grep "bluez_card" | cut -f2)
if [ -n "$CARD" ]; then
    pactl set-card-profile $CARD a2dp-sink
    echo "카드 인식 성공: $CARD"
else
    echo "여전히 카드가 안 보임. 블루투스 껐다 켜는 중..."
    bluetoothctl power off && sleep 1 && bluetoothctl power on
fi

echo "완료. 현재 시각: $(date +'%H:%M:%S')"
