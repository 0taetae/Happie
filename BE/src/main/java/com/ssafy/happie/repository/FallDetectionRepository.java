package com.ssafy.happie.repository;

import com.ssafy.happie.entity.FallDetection;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FallDetectionRepository extends JpaRepository<FallDetection, Integer> {
}