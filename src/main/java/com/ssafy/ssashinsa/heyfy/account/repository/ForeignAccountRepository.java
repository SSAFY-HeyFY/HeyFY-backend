package com.ssafy.ssashinsa.heyfy.account.repository;

import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import java.util.Optional;

public interface ForeignAccountRepository extends JpaRepository<ForeignAccount, Long> {
    Optional<ForeignAccount> findByAccountNo(String accountNo);

    Optional<ForeignAccount> findByUser(Users user);


    Optional<ForeignAccount> findByUserAndAccountNo(Users user, String accountNo);


    @Query("""
            SELECT fa
            FROM Users u
            JOIN u.foreignAccount fa
            WHERE LOWER(u.email) = LOWER(:email)
            """)
    Optional<ForeignAccount> findForeignAccountByUserEmail(String email);

    boolean existsByUser(Users user);


    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query("delete from ForeignAccount a where a.user = :user")
    void deleteByUser(Users user);
}

