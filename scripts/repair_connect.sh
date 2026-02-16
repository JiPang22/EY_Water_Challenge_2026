#!/bin/bash
MAC="68:52:10:5C:26:E7"

# 1. 백그라운드 스캔 시작 (장치 검색)
echo "블루투스 스캔 시작... 스피커바를 페어링 모드(BT 깜빡임)로 설정하세요."
bluetoothctl scan on > /dev/null 2>&1 &
SCAN_PID=$!

# 2. 장치 검색 대기 (15초)
for i in {1..15}; do
    echo -n "."
    sleep 1
done
echo ""

# 3. 신뢰 및 페어링 시도
echo "장치 연결 시도 중..."
bluetoothctl trust $MAC
bluetoothctl pair $MAC
sleep 3
bluetoothctl connect $MAC

# 4. 스캔 프로세스 정리
kill $SCAN_PID 2>/dev/null

# 5. 오디오 프로필 확인
sleep 2
pactl list sinks short | grep bluez

echo "설정 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+5 seconds' +'%H:%M:%S')"
