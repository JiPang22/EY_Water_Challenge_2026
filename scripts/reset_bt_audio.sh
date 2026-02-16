#!/bin/bash
# 1. 블루투스 페어링 정보 강제 삭제 (스피커바 주소를 몰라도 전체 초기화)
# 연결된 모든 블루투스 장치 캐시 제거
sudo rm -rf /var/lib/bluetooth/*

# 2. Pipewire 및 블루투스 관련 패키지 재설정
sudo apt-get install --reinstall -y bluez pipewire-audio-client-libraries libspa-0.2-bluetooth

# 3. 블루투스 오디오 프로필 강제 활성화 (AutoEnable)
sudo sed -i 's/^#\?AutoEnable.*/AutoEnable = true/' /etc/bluetooth/main.conf

# 4. 서비스 정지 -> 모듈 언로드 -> 재시작
systemctl --user stop wireplumber pipewire pipewire-pulse
sudo modprobe -r btusb
sudo modprobe btusb
sudo systemctl restart bluetooth
systemctl --user start pipewire pipewire-pulse wireplumber

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+10 seconds' +'%H:%M:%S')"
