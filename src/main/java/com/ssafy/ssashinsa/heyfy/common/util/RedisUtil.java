package com.ssafy.ssashinsa.heyfy.common.util;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class RedisUtil {
    private final StringRedisTemplate redisTemplate;

    @Value("${spring.jwt.refresh-expiration}")
    private long refreshExpirationMs;

    public void setRefreshToken(String key, String value) {
        long timeoutSeconds = refreshExpirationMs / 1000;
        try {
            redisTemplate.opsForValue().set(key, value, timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set refresh token in Redis", e);
        }
    }

    public String getRefreshToken(String key) {
        return redisTemplate.opsForValue().get(key);
    }

    public void deleteRefreshToken(String key) {
        try {
            redisTemplate.delete(key);
        } catch (Exception e) {
            throw new RuntimeException("Failed to delete refresh token from Redis", e);
        }
    }
}
