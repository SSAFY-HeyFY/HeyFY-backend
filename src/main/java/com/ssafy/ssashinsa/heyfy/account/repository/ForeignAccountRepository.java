package com.ssafy.ssashinsa.heyfy.account.repository;

import com.github.f4b6a3.ulid.Ulid;
import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ForeignAccountRepository extends JpaRepository<Account, Long> {
    Optional<ForeignAccount> findByAccountNo(String accountNo);

    Optional<ForeignAccount> findByUser(Users user);

}

