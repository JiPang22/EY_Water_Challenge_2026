#!/bin/bash
# 1. 블루투스 오디오 노드 ID 추출
NODE_ID=$(pw-dump | jq '.[] | select(.info?.props?."node.name"? // "" | contains("bluez_output")) | .id' | head -n 1)

if [ -z "$NODE_ID" ]; then
    echo "에러: 블루투스 출력 노드를 찾을 수 없음."
    exit 1
fi

# 2. 런타임에서 절전(Suspend) 및 일시정지(Pause) 즉시 비활성화
pw-metadata -n objects $NODE_ID session.suspend-on-idle false 2>/dev/null
pw-metadata -n objects $NODE_ID node.pause-on-idle false 2>/dev/null

# 3. 적용 결과 확인
echo "--- 현재 노드($NODE_ID) 설정 상태 ---"
pw-metadata -n objects $NODE_ID | grep -E "suspend|pause"

echo "검증 완료. 현재 시각: $(date +'%H:%M:%S')"
echo "예상 종료 시간: $(date -d '+2 seconds' +'%H:%M:%S')"
