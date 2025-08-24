package com.ssafy.ssashinsa.heyfy.user.repository;


import com.github.f4b6a3.ulid.Ulid;
import com.ssafy.ssashinsa.heyfy.account.dto.AccountPairDto;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import feign.Param;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Optional;

public interface UserRepository extends JpaRepository<Users, Ulid> {
    Optional<Users> findByStudentId(String username);
    Optional<Users> findByEmail(String email);

    @Query("""
            SELECT NEW com.ssafy.ssashinsa.heyfy.account.dto.AccountPairDto(a, fa)
            FROM Users u
            JOIN u.account a
            JOIN u.foreignAccount fa
            WHERE LOWER(u.email) = LOWER(:email)
            """)
    Optional<AccountPairDto> findAccountsByUserEmail(@Param("email") String email);

    @Query("""
            SELECT u FROM Users u
            JOIN FETCH u.account a
            JOIN FETCH u.foreignAccount fa
            WHERE LOWER(u.studentId) = LOWER(:studentId)
            """)
    Optional<Users> findUserWithAccountsByStudentId(@Param("studentId") String studentId);
}
