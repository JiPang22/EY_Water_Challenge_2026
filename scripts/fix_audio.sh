#!/bin/bash
# 1. 블루투스 절전 모드 비활성화(IdleTimeout=0)
sudo sed -i 's/^#\?IdleTimeout.*/IdleTimeout = 0/' /etc/bluetooth/main.conf

# 2. Wi-Fi 및 블루투스 공존(Coexistence) 활성화
echo "options iwlwifi bt_coex_active=1" | sudo tee /etc/modprobe.d/iwlwifi.conf > /dev/null

# 3. Pipewire 샘플 레이트(Sampling Rate) 48kHz 고정
if [ -f /etc/pipewire/pipewire.conf ]; then
    sudo sed -i 's/^#\?default.clock.rate.*/default.clock.rate = 48000/' /etc/pipewire/pipewire.conf
fi

# 4. 서비스 재시작 및 상태 확인
sudo systemctl restart bluetooth
systemctl --user restart pipewire pipewire-pulse

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
