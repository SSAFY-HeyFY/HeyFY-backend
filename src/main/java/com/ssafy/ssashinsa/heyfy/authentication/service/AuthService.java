package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.authentication.dto.*;
import com.ssafy.ssashinsa.heyfy.authentication.entity.Users;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.JwtTokenProvider;
import com.ssafy.ssashinsa.heyfy.authentication.repository.UserRepository;
import com.ssafy.ssashinsa.heyfy.authentication.util.RedisUtil;
import com.ssafy.ssashinsa.heyfy.authentication.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

@Service
@RequiredArgsConstructor
public class AuthService {
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider jwtTokenProvider;
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final RedisUtil redisUtil;

    public SignInSuccessDto signIn(SignInDto signInDto) {
        try {
            UsernamePasswordAuthenticationToken authenticationToken =
                    new UsernamePasswordAuthenticationToken(signInDto.getUsername(), signInDto.getPassword());

            Authentication authentication = authenticationManager.authenticate(authenticationToken);

            // jti를 통해서 액세스 토큰-리프레쉬 토큰 쌍이 올바른지 체크
            String jti = UUID.randomUUID().toString();
            String accessToken = jwtTokenProvider.createAccessToken(authentication, jti);
            String refreshToken = jwtTokenProvider.createRefreshToken(authentication, jti);

            redisUtil.deleteRefreshToken(signInDto.getUsername());
            redisUtil.setRefreshToken(signInDto.getUsername(), refreshToken);

            return new SignInSuccessDto(accessToken, refreshToken);
        } catch (BadCredentialsException e) {
            throw new CustomException(ErrorCode.LOGIN_FAILED);
        }
    }

    @Transactional
    public SignUpSuccessDto signUp(SignUpDto signUpDto) {
        userRepository.findByUsername(signUpDto.getUsername()).ifPresent(user -> {
            throw new CustomException(ErrorCode.EXIST_USER_NAME);
        });

        userRepository.findByEmail(signUpDto.getEmail()).ifPresent(user -> {
            throw new CustomException(ErrorCode.EXIST_EMAIL);
        });

        String encodedPassword = passwordEncoder.encode(signUpDto.getPassword());

        Users user = Users.builder()
                .username(signUpDto.getUsername())
                .password(encodedPassword)
                .name(signUpDto.getName())
                .email(signUpDto.getEmail())
                .language(signUpDto.getLanguage())
                .univName(signUpDto.getUnivName())
                .build();

        Users savedUser = userRepository.save(user);

        return new SignUpSuccessDto("회원가입이 성공적으로 완료되었습니다.", savedUser.getUsername());
    }

    // 리프레쉬 토큰 재발급
    public TokenDto refreshAccessToken(String authorizationHeader, String refreshToken) {

        validateRefreshToken(refreshToken);

        String refreshTokenUsername = jwtTokenProvider.getUsernameFromToken(refreshToken);
        validateRefreshTokenInRedis(refreshTokenUsername, refreshToken);

        validateAccessTokenAndUserMatch(authorizationHeader, refreshToken);



        Authentication authentication = new UsernamePasswordAuthenticationToken(refreshTokenUsername, null, null);

        String newJti = UUID.randomUUID().toString();

        String newAccessToken = jwtTokenProvider.createAccessToken(authentication, newJti);
        String newRefreshToken = jwtTokenProvider.createRefreshToken(authentication, newJti);

        redisUtil.deleteRefreshToken(refreshTokenUsername);
        redisUtil.setRefreshToken(refreshTokenUsername, newRefreshToken);

        return new TokenDto(newAccessToken, newRefreshToken);
    }
    // RefreshToken 유효성 검증
    private void validateRefreshToken(String refreshToken) {
        if (refreshToken == null || refreshToken.isEmpty()) {
            throw new CustomException(ErrorCode.MISSING_REFRESH_TOKEN);
        }
        try {
            jwtTokenProvider.validateToken(refreshToken);
        } catch (CustomException e) {
            throw new CustomException(ErrorCode.INVALID_REFRESH_TOKEN); // 안드로이드 상에서는 INVALID_REFRESH_TOKEN 받았을때 재 로그인 하도록 로직 구분
        }
    }

    // RefreshToken이 redis에 저장되어 있는지 검증
    private void validateRefreshTokenInRedis(String refreshTokenUsername, String refreshToken) {
        String redisRefreshToken = redisUtil.getRefreshToken(refreshTokenUsername);
        if (redisRefreshToken == null || !redisRefreshToken.equals(refreshToken)) {
            throw new CustomException(ErrorCode.INVALID_REFRESH_TOKEN);
        }
    }

    // AccessToken 유효성 검증, AccessToken 만료여부 확인, AccessToken과 refreshToken 비교
    private void validateAccessTokenAndUserMatch(String authorizationHeader, String refreshToken) {
        if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer ")) {
            throw new CustomException(ErrorCode.MISSING_ACCESS_TOKEN);
        }
        String accessToken = authorizationHeader.substring(7);
        try {
            jwtTokenProvider.validateToken(accessToken);

            /*
            //테스트가 까다로워지므로 주석처리. 후일 필요하다면 주석 제거해서 사용
            // 보안 강화: 유효한 Access Token으로 갱신 요청 시 Refresh Token 무효화(악의적인 사용자에게 정보를 주지 않기 위함)
            redisUtil.deleteRefreshToken(jwtTokenProvider.getUsernameFromToken(refreshToken));
            */

            throw new CustomException(ErrorCode.NOT_EXPIRED_TOKEN);
        } catch (CustomException e) {
            if (!e.getErrorCode().equals(ErrorCode.EXPIRED_TOKEN)) {
                throw e;
            }
            String accessTokenJti = jwtTokenProvider.getJtiFromToken(accessToken);
            String refreshTokenJti = jwtTokenProvider.getJtiFromToken(refreshToken);

            if (!accessTokenJti.equals(refreshTokenJti)) {
                throw new CustomException(ErrorCode.TOKEN_PAIR_MISMATCH);
            }
        }
    }

    public Users getCurrentUser() {
        String username = SecurityUtil.getCurrentUsername();
        if (username == null) {
            throw new CustomException(ErrorCode.UNAUTHORIZED);
        }
        return userRepository.findByUsername(username)
                .orElseThrow(() -> new CustomException(ErrorCode.UNAUTHORIZED));
    }

    public String getCurrentUserKey() {
        return getCurrentUser().getUserKey();
    }

}
