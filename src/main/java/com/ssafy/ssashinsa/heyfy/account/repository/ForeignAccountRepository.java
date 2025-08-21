package com.ssafy.ssashinsa.heyfy.account.repository;

import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Optional;

public interface ForeignAccountRepository extends JpaRepository<ForeignAccount, Long> {
    Optional<ForeignAccount> findByAccountNo(String accountNo);

    Optional<ForeignAccount> findByUser(Users user);

    @Query("""
            SELECT fa
            FROM Users u
            JOIN u.foreignAccount fa
            WHERE LOWER(u.email) = LOWER(:email)
            """)
    Optional<ForeignAccount> findForeignAccountByUserEmail(String email);
}

