package com.ssafy.happie.controller;

import com.ssafy.happie.dto.FallDetectionRequestDto;
import com.ssafy.happie.entity.FallDetection;
import com.ssafy.happie.service.FallDetectionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "fallDetection")
@Slf4j
@RestController
@RequestMapping("/api/fall-detection")
@RequiredArgsConstructor
public class FallDetectionController {
    private final FallDetectionService fallDetectionService;

    @PostMapping("/confirm")
    @Operation(summary = "낙상 확인", description = "true 받으면 됨")
    public String checkFallDetection(@RequestBody FallDetectionRequestDto requestDto) {
        return fallDetectionService.checkFallDetection(requestDto);
    }
}