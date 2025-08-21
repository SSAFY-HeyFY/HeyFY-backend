package com.ssafy.ssashinsa.heyfy.account.repository;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Optional;

public interface AccountRepository extends JpaRepository<Account, Long> {
    Optional<Account> findByAccountNo(String accountNo);

    Optional<Account> findByUser(Users user);


    Optional<Account> findByUserAndAccountNo(Users user, String accountNo);


    @Query("""
            SELECT a
            FROM Users u
            JOIN u.account a
            WHERE LOWER(u.email) = LOWER(:email)
            """)
    Optional<Account> findAccountByUserEmail(String email);

}
