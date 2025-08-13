package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInSuccessDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.TokenDto;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.JwtTokenProvider;
import com.ssafy.ssashinsa.heyfy.authentication.util.RedisUtil;
import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AuthService {
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider jwtTokenProvider;
    private final RedisUtil redisUtil;

    public SignInSuccessDto signIn(SignInDto signInDto) {
        try {
            UsernamePasswordAuthenticationToken authenticationToken =
                    new UsernamePasswordAuthenticationToken(signInDto.getUsername(), signInDto.getPassword());

            Authentication authentication = authenticationManager.authenticate(authenticationToken);
            String accessToken = jwtTokenProvider.createAccessToken(authentication);
            String refreshToken = jwtTokenProvider.createRefreshToken(authentication);

            redisUtil.deleteRefreshToken(signInDto.getUsername());
            redisUtil.setRefreshToken(signInDto.getUsername(), refreshToken);

            return new SignInSuccessDto(accessToken, refreshToken);
        } catch (BadCredentialsException e) {
            throw new CustomException(ErrorCode.LOGIN_FAILED);
        }
    }

    public TokenDto refreshAccessToken(String authorizationHeader, String refreshToken) {
        // 1. RefreshToken 유효성 검증
        validateRefreshToken(refreshToken);
        String refreshTokenUsername = jwtTokenProvider.getUsernameFromToken(refreshToken);

        validateAccessTokenAndUserMatch(authorizationHeader, refreshTokenUsername);

        validateRefreshTokenInRedis(refreshTokenUsername, refreshToken);

        Authentication authentication = new UsernamePasswordAuthenticationToken(refreshTokenUsername, null, null);
        String newAccessToken = jwtTokenProvider.createAccessToken(authentication);
        String newRefreshToken = jwtTokenProvider.createRefreshToken(authentication);

        redisUtil.deleteRefreshToken(refreshTokenUsername);
        redisUtil.setRefreshToken(refreshTokenUsername, newRefreshToken);

        return new TokenDto(newAccessToken, newRefreshToken);
    }

    private void validateRefreshToken(String refreshToken) {
        if (refreshToken == null || refreshToken.isEmpty()) {
            throw new CustomException(ErrorCode.MISSING_REFRESH_TOKEN);
        }
        try {
            jwtTokenProvider.validateToken(refreshToken);
        } catch (CustomException e) {
            throw new CustomException(ErrorCode.INVALID_REFRESH_TOKEN);
        }
    }

    private void validateAccessTokenAndUserMatch(String authorizationHeader, String refreshTokenUsername) {
        if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer ")) {
            throw new CustomException(ErrorCode.MISSING_ACCESS_TOKEN);
        }
        String accessToken = authorizationHeader.substring(7);
        try {
            jwtTokenProvider.validateToken(accessToken);
            // 보안 강화: 유효한 Access Token으로 갱신 요청 시 Refresh Token 무효화
            redisUtil.deleteRefreshToken(refreshTokenUsername);
            throw new CustomException(ErrorCode.NOT_EXPIRED_TOKEN);
        } catch (CustomException e) {
            if (!e.getErrorCode().equals(ErrorCode.EXPIRED_TOKEN)) {
                throw e;
            }
            String accessTokenUsername = jwtTokenProvider.getUsernameFromToken(accessToken);
            if (!accessTokenUsername.equals(refreshTokenUsername)) {
                throw new CustomException(ErrorCode.UNAUTHORIZED);
            }
        }
    }

    private void validateRefreshTokenInRedis(String refreshTokenUsername, String refreshToken) {
        String redisRefreshToken = redisUtil.getRefreshToken(refreshTokenUsername);
        if (redisRefreshToken == null || !redisRefreshToken.equals(refreshToken)) {
            throw new CustomException(ErrorCode.UNAUTHORIZED);
        }
    }






}
