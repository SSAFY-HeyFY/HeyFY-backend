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

    private static final String REFRESH_TOKEN_PREFIX = "refresh:";
    private static final String TXN_AUTH_TOKEN_PREFIX = "txnAuth:";

    public void setRefreshToken(String key, String value) {
        long timeoutSeconds = refreshExpirationMs / 1000;
        redisTemplate.opsForValue().set(REFRESH_TOKEN_PREFIX + key, value, timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS);
    }

    public String getRefreshToken(String key) {
        return redisTemplate.opsForValue().get(REFRESH_TOKEN_PREFIX + key);
    }

    public void deleteRefreshToken(String key) {
        redisTemplate.delete(REFRESH_TOKEN_PREFIX + key);
    }

    public void setTxnAuthToken(String key, String value, long expiration, java.util.concurrent.TimeUnit timeUnit) {
        redisTemplate.opsForValue().set(TXN_AUTH_TOKEN_PREFIX + key, value, expiration, timeUnit);
    }

    public String getTxnAuthToken(String key) {
        return redisTemplate.opsForValue().get(TXN_AUTH_TOKEN_PREFIX + key);
    }

    public void deleteTxnAuthToken(String key) {
        redisTemplate.delete(TXN_AUTH_TOKEN_PREFIX + key);
    }
}
