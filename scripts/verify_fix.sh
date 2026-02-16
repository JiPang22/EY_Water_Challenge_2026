#!/bin/bash
# 1. jq 필터 보정: 필드 존재 여부 체크 추가
pw-dump | jq '.[] | select(.info?.props?."node.name"? // "" | contains("bluez_output")) | .info.props' | grep -E "suspend|pause"

# 2. WirePlumber 런타임 값 직접 조회
wpctl inspect $(pw-link -i | grep bluez_output | head -n 1 | awk '{print $1}') 2>/dev/null | grep -E "suspend|pause"

echo "검증 완료. 현재 시각: $(date +'%H:%M:%S')"
