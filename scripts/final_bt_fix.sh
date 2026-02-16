#!/bin/bash
# 1. 블루투스 오디오 프로필 모드 강제 전환 (GATT -> Breed)
sudo sed -i 's/^#\?ControllerMode.*/ControllerMode = bredr/' /etc/bluetooth/main.conf

# 2. PipeWire 블루투스 모듈 수동 설치 및 링크 확인
sudo apt-get install -y pipewire-audio-client-libraries libspa-0.2-bluetooth pipewire-pulse
sudo apt-get remove -y pulseaudio-module-bluetooth # 충돌 방지

# 3. 서비스 환경 변수 초기화 및 세션 재시작
systemctl --user stop wireplumber pipewire pipewire-pulse
rm -rf ~/.local/state/wireplumber/*
sudo systemctl restart bluetooth

# 4. 서비스 다시 시작
systemctl --user start pipewire pipewire-pulse wireplumber

# 5. 장치 페어링 완전 초기화
MAC_ADDR="68:52:10:5C:26:E7"
bluetoothctl remove $MAC_ADDR
echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+15 seconds' +'%H:%M:%S')"
