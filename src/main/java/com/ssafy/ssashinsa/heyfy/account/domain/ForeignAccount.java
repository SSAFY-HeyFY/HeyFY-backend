package com.ssafy.ssashinsa.heyfy.account.domain;

import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "foreign_account")
@Getter
@Builder
@NoArgsConstructor(access = AccessLevel.PROTECTED) // JPA를 위한 기본 생성자
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class ForeignAccount {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "foreign_account_id")
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private Users user;

    @Column(name = "account_no", nullable = false, unique = true)
    private String accountNo;

    @Column(name = "currency", nullable = false) // 💡 currency 필드 추가
    private String currency;
}
