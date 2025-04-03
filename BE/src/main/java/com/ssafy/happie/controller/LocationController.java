package com.ssafy.happie.controller;

import com.ssafy.happie.dto.LocationRequestDto;
import com.ssafy.happie.dto.LocationResponseDto;
import com.ssafy.happie.service.LocationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

@Tag(name = "location")
@Slf4j
@RestController
@RequestMapping("/api/location")
@RequiredArgsConstructor
public class LocationController {
    private final LocationService locationService;

    @PostMapping("/name")
    @Operation(summary = "시설 이름 받아오기", description = "ex.501호실 (시설명만 적어주기)")
    public LocationResponseDto getLocationCoordinates(@RequestBody LocationRequestDto locationRequestDto) {
        return locationService.getLocationCoordinates(locationRequestDto);
    }
}
