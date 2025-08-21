package com.ssafy.ssashinsa.heyfy.account.repository;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface AccountRepository extends JpaRepository<Account, Long> {
    Optional<Account> findByAccountNo(String accountNo);

    Optional<Account> findByUser(Users user);

    Optional<Account> findByUserAndAccountNo(Users user, String accountNo);

}
