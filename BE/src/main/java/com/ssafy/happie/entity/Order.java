package com.ssafy.happie.entity;

import jakarta.persistence.*;
import lombok.NoArgsConstructor;

@Entity(name = "orders")
@NoArgsConstructor
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column
    private String robot;

    @Column
    private String place;

    @Column
    private String todo;

    @Column
    private String state = "대기";
}
