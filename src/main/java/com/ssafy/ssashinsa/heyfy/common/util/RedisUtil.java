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

    @Value("${spring.data.redis.sid-expiration}")
    private long sidExpirationSeconds;

    private static final String REFRESH_TOKEN_PREFIX = "refresh:";
    private static final String TXN_AUTH_TOKEN_PREFIX = "txnAuth:";
    private static final String SID_PREFIX = "sid:";

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

    public void setSid(String sid, String userId) {
        redisTemplate.opsForValue().set(SID_PREFIX + sid, userId, sidExpirationSeconds, java.util.concurrent.TimeUnit.SECONDS);
    }

    public String getSid(String sid) {
        return redisTemplate.opsForValue().get(SID_PREFIX + sid);
    }

    public void updateSidExpiration(String sid) {
        // Redis에 sid가 존재할 경우에만 만료 시간 갱신
        if (getSid(sid) != null) {
            redisTemplate.expire(SID_PREFIX + sid, sidExpirationSeconds, java.util.concurrent.TimeUnit.SECONDS);
        }
    }
}
