#!/bin/bash
# 1. 하드웨어 차단(Soft block) 해제
sudo rfkill unblock bluetooth

# 2. 블루투스 커널 모듈 강제 초기화
sudo modprobe -r btusb
sudo modprobe btusb

# 3. 블루투스 서비스 강제 재시작 및 전원 켜기
sudo systemctl restart bluetooth
sleep 2
bluetoothctl power on

# 4. 장치 검색 및 페어링 초기화 시도
MAC_ADDR="68:52:10:5C:26:E7"
bluetoothctl remove $MAC_ADDR 2>/dev/null
bluetoothctl scan on &
SCAN_PID=$!
sleep 5
kill $SCAN_PID

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+10 seconds' +'%H:%M:%S')"
