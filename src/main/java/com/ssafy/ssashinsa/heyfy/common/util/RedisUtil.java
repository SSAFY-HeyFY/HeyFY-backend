package com.ssafy.ssashinsa.heyfy.common.util;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;

@Component
@RequiredArgsConstructor
public class RedisUtil {
    private final StringRedisTemplate redisTemplate;

    @Value("${spring.jwt.refresh-expiration}")
    private long refreshExpirationMs;

    @Value("${spring.jwt.access-expiration}")
    private long accessExpirationMs;

    @Value("${spring.data.redis.sid-expiration}")
    private long sidExpirationSeconds;
    private static final String PIN_FAILED_ATTEMPTS_PREFIX = "pinFailed:";
    private static final String AT_BLACKLIST_PREFIX = "atBlacklist:";
    private static final String REFRESH_TOKEN_PREFIX = "refresh:";
    private static final String SID_PREFIX = "sid:";
    private static final String TEMP_LOCK_PREFIX = "temp:";
    private static final String TRADE_PIN_FAILED_PREFIX = "tradePinFailed:";
    private static final String TRADE_PIN_LOCK_PREFIX = "tradePinLock:";

    public void setRefreshToken(String key, String value) {
        long timeoutSeconds = refreshExpirationMs;
        redisTemplate.opsForValue().set(REFRESH_TOKEN_PREFIX + key, value, timeoutSeconds, TimeUnit.SECONDS);
    }

    public String getRefreshToken(String key) {
        return redisTemplate.opsForValue().get(REFRESH_TOKEN_PREFIX + key);
    }

    public void deleteRefreshToken(String key) {
        redisTemplate.delete(REFRESH_TOKEN_PREFIX + key);
    }

    public void setSid(String sid, String userId) {
        redisTemplate.opsForValue().set(SID_PREFIX + userId, sid, sidExpirationSeconds, TimeUnit.SECONDS);
    }

    // SID를 가져올 때 userId를 사용
    public String getSidByUserId(String userId) {
        return redisTemplate.opsForValue().get(SID_PREFIX + userId);
    }

    public void updateSidExpiration(String userId) {
        // Redis에 해당 사용자 ID로 SID가 존재할 경우에만 만료 시간 갱신
        if (redisTemplate.hasKey(SID_PREFIX + userId)) {
            redisTemplate.expire(SID_PREFIX + userId, sidExpirationSeconds, TimeUnit.SECONDS);
        }
    }

    public boolean setTokenRefreshLock(String jti, long timeout, TimeUnit timeUnit) {
        return Boolean.TRUE.equals(redisTemplate.opsForValue().setIfAbsent(TEMP_LOCK_PREFIX + jti, "locked", timeout, timeUnit));
    }

    public void deleteTokenRefreshLock(String jti) {
        redisTemplate.delete(TEMP_LOCK_PREFIX + jti);
    }


    public long incrementPinFailedAttempts(String studentId) {
        String key = PIN_FAILED_ATTEMPTS_PREFIX + studentId;
        Long count = redisTemplate.opsForValue().increment(key);
        if (count != null && count == 1) {
            redisTemplate.expire(key, accessExpirationMs, TimeUnit.SECONDS);
        }
        return count != null ? count : 0;
    }

    public void deletePinFailedAttempts(String studentId) {
        redisTemplate.delete(PIN_FAILED_ATTEMPTS_PREFIX + studentId);
    }

    public void setAccessTokenBlacklist(String jti) {
        redisTemplate.opsForValue().set(AT_BLACKLIST_PREFIX + jti, "blacklisted", accessExpirationMs, TimeUnit.SECONDS);
    }

    public boolean isAccessTokenBlacklisted(String jti) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(AT_BLACKLIST_PREFIX + jti));
    }

    public long incrementTradePinFailedAttempts(String studentId) {
        String key = TRADE_PIN_FAILED_PREFIX + studentId;
        Long count = redisTemplate.opsForValue().increment(key);
        if (count != null && count == 1) {
            redisTemplate.expire(key, accessExpirationMs, TimeUnit.SECONDS);
        }
        return count != null ? count : 0;
    }

    public void deleteTradePinFailedAttempts(String studentId) {
        redisTemplate.delete(TRADE_PIN_FAILED_PREFIX + studentId);
    }

    public void setTradePinLock(String studentId) {
        redisTemplate.opsForValue().set(TRADE_PIN_LOCK_PREFIX + studentId, "locked", 30L, TimeUnit.SECONDS);
    }

    public boolean isTradePinLocked(String studentId) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(TRADE_PIN_LOCK_PREFIX + studentId));
    }

    public void deleteTradePinLock(String studentId) {
        redisTemplate.delete(TRADE_PIN_LOCK_PREFIX + studentId);
    }






}
