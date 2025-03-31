package com.ssafy.happie.entity;

import jakarta.persistence.*;

@Entity(name = "orders")
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column
    private String robot;

    @Column
    private String place;

    @Column
    private String todo;

    @Column
    private String state;
}
