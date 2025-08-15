package com.ssafy.ssashinsa.heyfy.authentication.repository;


import com.ssafy.ssashinsa.heyfy.authentication.entity.Users;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.UUID;

public interface UserRepository extends JpaRepository<Users, UUID> {
    Optional<Users> findByUsername(String username);
}
