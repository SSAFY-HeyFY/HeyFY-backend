package com.ssafy.ssashinsa.heyfy.common;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ErrorCode {
    INVALID_FIELD(HttpStatus.BAD_REQUEST, "잘못된 필드입니다."),
    UNAUTHORIZED(HttpStatus.UNAUTHORIZED,"인증되지 않은 사용자입니다"),
    INVALID_ACCESS_TOKEN(HttpStatus.UNAUTHORIZED, "유효하지 않은 액세스 토큰입니다."),
    INVALID_REFRESH_TOKEN(HttpStatus.UNAUTHORIZED, "유효하지 않은 리프레쉬 토큰입니다."),
    EXPIRED_TOKEN(HttpStatus.UNAUTHORIZED, "액세스 토큰이 만료되었습니다."),
    INVALID_SIGNATURE(HttpStatus.UNAUTHORIZED, "유효하지 않은 서명입니다."),
    LOGIN_FAILED(HttpStatus.UNAUTHORIZED, "아이디 또는 비밀번호가 틀립니다."),


    TOKEN_PAIR_MISMATCH(HttpStatus.BAD_REQUEST, "액세스 토큰과 리프레쉬 토큰이 매치되지 않습니다"),
    MISSING_ACCESS_TOKEN(HttpStatus.BAD_REQUEST, "액세스 토큰이 포함되지 않았습니다."),
    MISSING_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "리프레쉬 토큰이 포함되지 않았습니다"),
    NOT_EXPIRED_TOKEN(HttpStatus.BAD_REQUEST, "만료되지 않은 액세스 토큰입니다");

    private final HttpStatus httpStatus;
    private final String message;


}
