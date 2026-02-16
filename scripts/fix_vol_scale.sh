#!/bin/bash
# 1. 블루투스 하드웨어 볼륨 동기화 비활성화 (hw-volume = false)
# 슬라이더가 스피커 앰프를 직접 건드리지 않게 설정
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat << 'CONF' > ~/.config/wireplumber/wireplumber.conf.d/51-bluez-config.conf
monitor.bluez.rules = [
  {
    matches = [ { "device.name" = "~bluez_card.*" } ]
    actions = {
      update-props = {
        "bluez5.autoswitch-profile" = false
        "bluez5.hw-volume" = false
        "bluez5.a2dp.ldac.quality" = "hq"
      }
    }
  }
]
CONF

# 2. 변경 사항 적용을 위한 서비스 재시작
systemctl --user restart wireplumber pipewire pipewire-pulse

# 3. 안전을 위해 초기 소프트웨어 볼륨 30%로 시작
sleep 2
SINK_NAME=$(pactl list sinks short | grep "bluez_output" | cut -f2 | head -n 1)
if [ -n "$SINK_NAME" ]; then
    pactl set-sink-volume $SINK_NAME 30%
fi

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
