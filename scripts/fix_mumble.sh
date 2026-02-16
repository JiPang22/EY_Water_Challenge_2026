#!/bin/bash
# 1. PipeWire 유휴 상태 절전 모드 비활성화 설정 파일 생성
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat << 'CONF' > ~/.config/pipewire/pipewire.conf.d/99-stop-suspend.conf
monitor.rules = [
  {
    matches = [ { "node.name" = "~bluez_output.*" } ]
    actions = {
      update-props = {
        "session.suspend-on-idle" = false
        "node.pause-on-idle" = false
      }
    }
  }
]
CONF

# 2. WirePlumber 블루투스 하드웨어 절전 방지
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat << 'CONF' > ~/.config/wireplumber/wireplumber.conf.d/51-bluez-suspend.conf
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

# 3. 서비스 재시작 (설정 반영)
systemctl --user restart wireplumber pipewire pipewire-pulse

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
