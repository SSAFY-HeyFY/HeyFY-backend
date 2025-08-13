package com.ssafy.ssashinsa.heyfy.authentication.jwt;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.exceptions.JWTVerificationException;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.stereotype.Component;

import java.util.Date;
import java.util.stream.Collectors;

@Component
public class JwtTokenProvider {

    @Value("${spring.jwt.secret}")
    private String secretKey;

    @Value("${spring.jwt.access-expiration}")
    private long accessExpiration;

    public String createAccessToken(Authentication authentication) {
        String username = authentication.getName();
        String roles = authentication.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.joining(","));

        Date now = new Date();
        Date validity = new Date(now.getTime() + accessExpiration);

        return JWT.create()
                .withSubject(username)
                .withClaim("roles", roles)
                .withIssuedAt(now)
                .withExpiresAt(validity)
                .sign(Algorithm.HMAC256(secretKey));
    }


    public String getUsernameFromToken(String token) {
        try {
            DecodedJWT jwt = JWT.decode(token);
            return jwt.getSubject();
        } catch (JWTVerificationException e) {
            throw new CustomException(ErrorCode.INVALID_ACCESS_TOKEN);
        }
    }


    public void validateToken(String token) {
        try {
            JWT.require(Algorithm.HMAC256(secretKey)).build().verify(token);
        } catch (com.auth0.jwt.exceptions.TokenExpiredException e) { //만료된 토큰
            throw new CustomException(ErrorCode.EXPIRED_TOKEN);
        } catch (JWTVerificationException e) { //유효하지 않은 토큰
            throw new CustomException(ErrorCode.INVALID_ACCESS_TOKEN);
        }
    }
}