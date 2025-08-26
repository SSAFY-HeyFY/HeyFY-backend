package com.ssafy.ssashinsa.heyfy.fcm.repository;

import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface FcmTokenRepository extends JpaRepository<FcmToken, Long> {
    boolean existsByToken(String token);
    Optional<FcmToken> findByToken(String token);   // 추가
    void deleteByToken(String token);
}

