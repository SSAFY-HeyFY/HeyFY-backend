package com.ssafy.ssashinsa.heyfy.common.util;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Component
@RequiredArgsConstructor
public class RedisUtil {
    private final StringRedisTemplate redisTemplate;

    @Value("${spring.jwt.refresh-expiration}")
    private long refreshExpirationMs;

    public void setRefreshToken(String key, String value) {
        long timeoutSeconds = refreshExpirationMs / 1000;
        try {
            log.info("Setting refresh token in Redis with key: {}, value: {}, timeout: {} seconds", key, value, timeoutSeconds);
            redisTemplate.opsForValue().set(key, value, timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS);
        } catch (Exception e) {
            log.error("Error setting refresh token in Redis: {}", e.getMessage());
            throw new RuntimeException("Failed to set refresh token in Redis", e);
        }
    }

    public String getRefreshToken(String key) {
        return redisTemplate.opsForValue().get(key);
    }

    public void deleteRefreshToken(String key) {
        try {
            log.info("Deleting refresh token from Redis with key: {}", key);
            redisTemplate.delete(key);
        } catch (Exception e) {
            log.error("Error deleting refresh token from Redis: {}", e.getMessage());
            log.error(e.getStackTrace().toString());
            throw new RuntimeException("Failed to delete refresh token from Redis", e);
        }
    }
}
