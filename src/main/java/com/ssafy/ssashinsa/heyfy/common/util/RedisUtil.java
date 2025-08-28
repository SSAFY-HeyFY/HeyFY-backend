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

    @Value("${spring.data.redis.sid-expiration}")
    private long sidExpirationSeconds;


    @Value("${spring.data.redis.temp-access-token-timeout}")
    private long TEMP_ACCESS_TOKEN_TIMEOUT;

    private static final String REFRESH_TOKEN_PREFIX = "refresh:";
    private static final String TXN_AUTH_TOKEN_PREFIX = "txnAuth:";
    private static final String SID_PREFIX = "sid:";
    private static final String TEMP_LOCK_PREFIX = "temp:";

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

    public void setTxnAuthToken(String key, String value, long expiration, TimeUnit timeUnit) {
        redisTemplate.opsForValue().set(TXN_AUTH_TOKEN_PREFIX + key, value, expiration, timeUnit);
    }

    public String getTxnAuthToken(String key) {
        return redisTemplate.opsForValue().get(TXN_AUTH_TOKEN_PREFIX + key);
    }

    public void deleteTxnAuthToken(String key) {
        redisTemplate.delete(TXN_AUTH_TOKEN_PREFIX + key);
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

    public void setTempAccessToken(String jti, String accessToken) {
        redisTemplate.opsForValue().set(TEMP_LOCK_PREFIX + jti, accessToken, TEMP_ACCESS_TOKEN_TIMEOUT, TimeUnit.SECONDS);
    }

    public String getTempAccessToken(String jti) {
        return redisTemplate.opsForValue().get(TEMP_LOCK_PREFIX + jti);
    }

    public boolean hasTempLock(String userId) {
        return redisTemplate.hasKey(TEMP_LOCK_PREFIX + userId);
    }

    public void deleteTempLock(String userId) {
        redisTemplate.delete(TEMP_LOCK_PREFIX + userId);
    }

}
