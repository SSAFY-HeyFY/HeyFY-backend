package com.ssafy.ssashinsa.heyfy.authentication.exception;

import org.springframework.http.HttpStatus;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum AuthErrorCode implements ErrorCode {

    UNAUTHORIZED(HttpStatus.UNAUTHORIZED,"인증되지 않은 사용자입니다"),
    INVALID_SIGNATURE(HttpStatus.UNAUTHORIZED, "유효하지 않은 서명입니다."),
    INVALID_ACCESS_TOKEN(HttpStatus.BAD_REQUEST, "유효하지 않은 액세스 토큰입니다."),
    INVALID_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "유효하지 않은 리프레쉬 토큰입니다."),
    EXPIRED_TOKEN(HttpStatus.BAD_REQUEST, "액세스 토큰이 만료되었습니다."),
    EXPIRED_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "리프레쉬 토큰이 만료되었습니다."),
    LOGIN_FAILED(HttpStatus.BAD_REQUEST, "아이디 또는 비밀번호가 틀립니다."),

    TOKEN_PAIR_MISMATCH(HttpStatus.BAD_REQUEST, "액세스 토큰과 리프레쉬 토큰이 매치되지 않습니다"),
    MISSING_ACCESS_TOKEN(HttpStatus.BAD_REQUEST, "액세스 토큰이 포함되지 않았습니다."),
    MISSING_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "리프레쉬 토큰이 포함되지 않았습니다"),
    NOT_EXPIRED_TOKEN(HttpStatus.BAD_REQUEST, "만료되지 않은 액세스 토큰입니다"),

    EXIST_USER_NAME(HttpStatus.BAD_REQUEST, "이미 존재하는 유저 아이디입니다"),
    EXIST_EMAIL(HttpStatus.BAD_REQUEST,"이미 존재하는 이메일입니다" ),
    INVALID_PASSWORD_FORMAT(HttpStatus.BAD_REQUEST, "비밀번호 형식이 맞지 않습니다."),
    USER_NOT_FOUND(HttpStatus.BAD_REQUEST, "사용자를 찾을 수 없습니다.");

    private final HttpStatus httpStatus;
    private final String message;
}
