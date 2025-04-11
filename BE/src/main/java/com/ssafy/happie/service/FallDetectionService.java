package com.ssafy.happie.service;

import com.ssafy.happie.dto.FallDetectionRequestDto;
import com.ssafy.happie.entity.FallDetection;
import com.ssafy.happie.repository.FallDetectionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class FallDetectionService {
    private final FallDetectionRepository fallDetectionRepository;

    @Transactional
    public String checkFallDetection(FallDetectionRequestDto requestDto) {
        if (!requestDto.isConfirm()) {
            return "낙상 확인 요청이 아님";
        }

        FallDetection fall = fallDetectionRepository.findById(1)
                .orElseThrow(() -> new IllegalArgumentException("낙상 감지 데이터가 없음"));

        if (!fall.isDetected()) {
            return "낙상 없음";
        }

        fall.setDetected(false);
        return "낙상사고 확인 완료";
    }
}
