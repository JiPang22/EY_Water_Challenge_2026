#!/bin/bash
# 1. Pipewire 버퍼 사이즈 및 퀀텀(Quantum) 설정 최적화
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat << 'CONF' > ~/.config/pipewire/pipewire.conf.d/99-low-latency.conf
context.properties = {
    default.clock.quantum = 1024
    default.clock.min-quantum = 512
    default.clock.max-quantum = 2048
}
CONF

# 2. 블루투스 오토 스위칭 비활성화 (HSP/HFP 방지)
# 유튜브 감상 시 고음질(A2DP) 고정을 위해 마이크 프로필 진입 차단
sed -i 's/^#\?bluez5.autoswitch-profile.*/bluez5.autoswitch-profile = false/' /etc/pipewire/pipewire-pulse.conf 2>/dev/null

# 3. 블루투스 높은 우선순위 설정 (Fast Connect)
sudo sed -i 's/^#\?FastConnectable.*/FastConnectable = true/' /etc/bluetooth/main.conf

# 4. 서비스 재시작
sudo systemctl restart bluetooth
systemctl --user restart pipewire pipewire-pulse

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
