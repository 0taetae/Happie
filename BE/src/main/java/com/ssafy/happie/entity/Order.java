package com.ssafy.happie.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity(name = "orders")
@Getter
@Setter
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
    private double x;

    @Column
    private double y;

    @Column
    private String todo;

    @Column
    private String state = "대기";

    public Order(String robot, String place, String todo, double x, double y) {
        this.robot = robot;
        this.place = place;
        this.todo = todo;
        this.x = x;
        this.y = y;
    }
}
