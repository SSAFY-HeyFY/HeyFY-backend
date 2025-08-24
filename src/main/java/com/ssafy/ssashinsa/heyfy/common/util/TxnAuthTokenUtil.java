package com.ssafy.ssashinsa.heyfy.common.util;

import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.JwtTokenProvider;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;
import java.util.UUID;

@Component
@RequiredArgsConstructor
public class TxnAuthTokenUtil {

    private final JwtTokenProvider jwtTokenProvider;
    private final RedisUtil redisUtil;
    private final PasswordEncoder passwordEncoder;

    public String createTxnAuthToken(String studentId) {
        String jti = UUID.randomUUID().toString();
        String txnAuthToken = jwtTokenProvider.createTxnAuthToken(studentId, jti, 10L, TimeUnit.MINUTES);

        redisUtil.setTxnAuthToken(studentId, txnAuthToken, 10L, TimeUnit.MINUTES);

        return txnAuthToken;
    }

    public void verifySecondaryAuth(String studentId, String pinNumber, Users user, String txnAuthToken) {
        if (txnAuthToken == null || txnAuthToken.isEmpty()) {
            throw new CustomException(AuthErrorCode.MISSING_TXN_AUTH_TOKEN);
        }

        try {

            jwtTokenProvider.validateToken(txnAuthToken);
        } catch (CustomException e) {
            if (e.getErrorCode().equals(AuthErrorCode.EXPIRED_TOKEN)) {
                throw new CustomException(AuthErrorCode.EXPIRED_TXN_AUTH_TOKEN);
            }
            throw e;
        }

        String redisToken = redisUtil.getTxnAuthToken(studentId);
        if (redisToken == null || !redisToken.equals(txnAuthToken)) {
            throw new CustomException(AuthErrorCode.INVALID_TXN_AUTH_TOKEN);
        }

        if (!passwordEncoder.matches(pinNumber, user.getPinNumber())) {
            throw new CustomException(AuthErrorCode.INVALID_PIN_NUMBER);
        }

        redisUtil.deleteTxnAuthToken(studentId);
    }
}