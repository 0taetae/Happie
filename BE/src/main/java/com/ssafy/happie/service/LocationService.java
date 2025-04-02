package com.ssafy.happie.service;

import com.ssafy.happie.dto.LocationResponseDto;
import com.ssafy.happie.entity.Location;
import com.ssafy.happie.repository.LocationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class LocationService {
    private final LocationRepository locationRepository;

    public LocationResponseDto getLocationCoordinates(String name) {
        return locationRepository.findByName(name)
                .map(location -> new LocationResponseDto(location.getX(), location.getY()))
                .orElse(null);
    }
}
