#!/bin/bash
# 1. 서비스 재시작 없이 런타임에서 퀀텀(Latency)만 조절
pw-metadata -n settings 0 default.clock.quantum 1024

# 2. 실시간 우선순위(Real-time Priority) 즉시 적용 확인
chrt -p 99 $(pgrep pipewire) 2>/dev/null

# 3. 블루투스 오디오 끊김 방지를 위한 비트레이트 협상 고정
# 서비스 재시작 없이 설정 파일만 덮어쓰기 (다음 연결 시 적용)
sudo sed -i 's/^#\?FastConnectable.*/FastConnectable = true/' /etc/bluetooth/main.conf

echo "런타임 설정 완료. 현재 시각: $(date +'%H:%M:%S')"
