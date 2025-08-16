package com.ssafy.ssashinsa.heyfy.authentication.entity;

import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@Builder
@Entity
@Table(name = "User")
@NoArgsConstructor
@AllArgsConstructor
// 스프링 User 클래스와 구분하기위해 Users 사용. 테이블명은 User 그대로 사용
public class Users {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(name = "user_id", length = 50)
    private String userId;

    @Column(name = "user_key")
    private String userKey;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "username", unique = true, nullable = false)
    private String username;

    @Column(name = "password", nullable = false)
    private String password;

    @Column(name = "email", unique = true, nullable = false)
    private String email;

    @Column(name = "language")
    private String language;

    @Column(name = "univ_name")
    private String univName;
}