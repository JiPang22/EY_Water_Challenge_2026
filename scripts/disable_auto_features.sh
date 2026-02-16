#!/bin/bash
# 1. WirePlumber: 블루투스 프로필 자동 전환 및 하드웨어 볼륨 연동 차단
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat << 'CONF' > ~/.config/wireplumber/wireplumber.conf.d/51-bluez-config.conf
monitor.bluez.rules = [
  {
    matches = [ { "device.name" = "~bluez_card.*" } ]
    actions = {
      update-props = {
        "bluez5.autoswitch-profile" = false
        "bluez5.hw-volume" = false
      }
    }
  }
]
CONF

# 2. Pipewire-Pulse: 플랫 볼륨(Flat Volumes) 비활성화
# 개별 앱 볼륨이 마스터 볼륨을 강제로 끌어올리는 것을 방지
mkdir -p ~/.config/pipewire/pipewire-pulse.conf.d
cat << 'CONF' > ~/.config/pipewire/pipewire-pulse.conf.d/99-disable-flat-volumes.conf
pulse.rules = [
    {
        actions = {
            quirks = [ "no-flat-volume" ]
        }
    }
]
CONF

# 3. 서비스 재시작으로 설정 적용
systemctl --user restart wireplumber pipewire pipewire-pulse

echo "자동 조절 기능 비활성화 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
