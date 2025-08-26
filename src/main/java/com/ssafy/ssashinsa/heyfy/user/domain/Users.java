package com.ssafy.ssashinsa.heyfy.user.domain;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
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
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long id;

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

    @Column(name = "pin_number")
    private String pinNumber;

    @Column(name = "email", unique = true, nullable = false)
    private String email;

    @Column(name = "language")
    private String language;

    @Column(name = "univ_name")
    private String univName;



    @OneToOne(mappedBy = "user", fetch = FetchType.LAZY)
    private Account account;

    @OneToOne(mappedBy = "user", fetch = FetchType.LAZY)
    private ForeignAccount foreignAccount;

    @PrePersist
    public void generateIds() {
        if (this.externalId == null) {
            this.externalId = UUID.randomUUID();
        }
    }
}