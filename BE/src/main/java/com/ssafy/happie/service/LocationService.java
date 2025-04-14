package com.ssafy.happie.service;

import com.ssafy.happie.config.MqttPublisher;
import com.ssafy.happie.dto.LocationRequestDto;
import com.ssafy.happie.dto.LocationResponseDto;
import com.ssafy.happie.repository.LocationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class LocationService {
    private final LocationRepository locationRepository;

    private final MqttPublisher mqttPublisher;

    @Transactional(readOnly = true)
    public LocationResponseDto getLocationCoordinates(LocationRequestDto locationRequestDto) {
        String name = locationRequestDto.getName();

        return locationRepository.findByName(name)
                .map(location -> {
                    double x = location.getX();
                    double y = location.getY();

//                    mqttPublisher.sendLocation(-1, x, y);

                    return new LocationResponseDto(-1, x, y);
                })
                .orElseThrow(() -> new IllegalArgumentException("해당 시설명을 찾을 수 없습니다: " + name));
    }
}
