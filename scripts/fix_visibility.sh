#!/bin/bash
# 1. 기존 블루투스 오디오 설정 캐시 삭제
rm -rf ~/.config/pulse
rm -rf ~/.local/state/wireplumber/*

# 2. 블루투스 오디오 프로필 강제 로드 설정 확인
sudo sed -i 's/^#\?Name = BlueZ/Name = BlueZ/' /etc/bluetooth/main.conf

# 3. 서비스 강제 재시작 (의존성 순서 준수)
systemctl --user stop wireplumber pipewire pipewire-pulse
sudo systemctl restart bluetooth
systemctl --user start pipewire pipewire-pulse wireplumber

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
