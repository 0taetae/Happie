package com.ssafy.happie.dto;

import com.ssafy.happie.entity.Order;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrderResponseDto {
    private Integer Id;

    private String robot;

    private String place;

    private String todo;

    private String state;
}
