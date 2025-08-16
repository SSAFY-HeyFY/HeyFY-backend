package com.ssafy.ssashinsa.heyfy.authentication.repository;


import com.github.f4b6a3.ulid.Ulid;
import com.ssafy.ssashinsa.heyfy.authentication.entity.Users;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<Users, Ulid> {
    Optional<Users> findByUsername(String username);
    Optional<Users> findByEmail(String email);
}
