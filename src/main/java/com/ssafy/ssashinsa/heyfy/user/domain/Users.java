package com.ssafy.ssashinsa.heyfy.user.domain;

import com.github.f4b6a3.ulid.Ulid;
import com.github.f4b6a3.ulid.UlidCreator;
import com.ssafy.ssashinsa.heyfy.user.ulid.UlidUserType;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.annotations.Type;
import org.hibernate.type.SqlTypes;

import java.util.UUID;

@Getter
@Setter
@Builder
@Entity
@Table(name = "Users")
@NoArgsConstructor
@AllArgsConstructor
// 스프링 User 클래스와 구분하기위해 Users 사용. 테이블명은 User 그대로 사용
public class Users {

    @Id
    @Type(UlidUserType.class)
    @Column(name = "user_id", length = 16)
    private Ulid userId;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(name = "external_id", unique = true, length = 36)
    private UUID externalId;

    @Column(name = "user_key")
    private String userKey;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "student_id", unique = true, nullable = false)
    private String studentId;

    @Column(name = "password", nullable = false)
    private String password;

    @Column(name = "email", unique = true, nullable = false)
    private String email;

    @Column(name = "language")
    private String language;

    @Column(name = "univ_name")
    private String univName;

    @PrePersist
    public void generateIds() {
        if (this.userId == null) {
            this.userId = UlidCreator.getMonotonicUlid();
        }
        if (this.externalId == null) {
            this.externalId = UUID.randomUUID();
        }
    }
}