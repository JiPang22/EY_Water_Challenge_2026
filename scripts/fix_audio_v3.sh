#!/bin/bash
# 1. WirePlumber 블루투스 고음질 코덱(LDAC/aptX) 우선순위 및 비트레이트 강제
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat << 'CONF' > ~/.config/wireplumber/wireplumber.conf.d/51-bluez-config.conf
monitor.bluez.rules = [
  {
    matches = [ { "device.name" = "~bluez_card.*" } ]
    actions = {
      update-props = {
        "bluez5.autoswitch-profile" = false
        "bluez5.hw-volume" = true
        "bluez5.a2dp.ldac.quality" = "hq"
      }
    }
  }
]
CONF

# 2. Pipewire 실시간 우선순위(Real-time Priority) 부여
sudo groupadd -r realtime 2>/dev/null
sudo usermod -aG realtime $USER
echo -e "@realtime - rtprio 99\n@realtime - memlock unlimited" | sudo tee /etc/security/limits.d/99-realtime.conf > /dev/null

# 3. 서비스 완전 재시작
systemctl --user restart wireplumber pipewire pipewire-pulse
sudo systemctl restart bluetooth

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
