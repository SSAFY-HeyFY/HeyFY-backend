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
    NOT_EXPIRED_TOKEN(HttpStatus.BAD_REQUEST, "만료되지 않은 액세스 토큰입니다"),
    EXIST_USER_NAME(HttpStatus.BAD_REQUEST, "이미 존재하는 유저 아이디입니다"),
    EXIST_EMAIL(HttpStatus.BAD_REQUEST,"이미 존재하는 이메일입니다" ),
    INVALID_PASSWORD_FORMAT(HttpStatus.BAD_REQUEST, "비밀번호 형식이 맞지 않습니다."),

    INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "에러가 발생했습니다"),
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "API 호출 에러"),
    API_USER_ALREADY_EXISTS(HttpStatus.INTERNAL_SERVER_ERROR, "신한 은행 API에 이미 등록된 이메일입니다"); //테스트용. 후에 배포 및 서비스시 에러 메세지가 사용자에게 노출 되지 않도록 수정

    private final HttpStatus httpStatus;
    private final String message;


}
