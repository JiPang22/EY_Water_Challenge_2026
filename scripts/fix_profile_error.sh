#!/bin/bash
# 1. 필수 블루투스 오디오 코덱 및 라이브러리 설치
sudo apt-get update
sudo apt-get install -y pipewire-audio-client-libraries libspa-0.2-bluetooth pipewire-pulse

# 2. 블루투스 모듈 강제 로드 설정 추가
sudo modprobe btusb
echo "load-module module-bluetooth-discover" | sudo tee -a /etc/pulse/default.pa > /dev/null

# 3. Pipewire 미디어 세션 및 서비스 초기화
systemctl --user stop wireplumber pipewire pipewire-pulse
rm -rf ~/.local/state/wireplumber/*
sudo systemctl restart bluetooth

# 4. 서비스 재시작
systemctl --user start pipewire pipewire-pulse wireplumber

# 5. 재연결 시도
MAC_ADDR="68:52:10:5C:26:E7"
sleep 2
bluetoothctl power off && sleep 1 && bluetoothctl power on
sleep 2
bluetoothctl connect $MAC_ADDR

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+10 seconds' +'%H:%M:%S')"
