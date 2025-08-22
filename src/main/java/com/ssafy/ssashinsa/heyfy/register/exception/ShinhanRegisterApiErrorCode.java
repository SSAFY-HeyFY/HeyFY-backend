package com.ssafy.ssashinsa.heyfy.register.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ShinhanRegisterApiErrorCode implements ErrorCode {
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "신한API 호출 에러"),
    MISSING_USER_KEY(HttpStatus.BAD_REQUEST, "유저키가 누락되었습니다."),
    USER_NOT_FOUND(HttpStatus.INTERNAL_SERVER_ERROR, "유저를 찾을 수 없습니다."),
    RESPONSE_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "신한API 응답 에러"),
    ACCOUNT_ALREADY_EXISTS(HttpStatus.BAD_REQUEST, "이미 계좌가 존재하는 유저입니다."),
    ACCOUNT_NOT_FOUND(HttpStatus.BAD_REQUEST, "유저와 연관된 계좌를 찾을 수 없습니다."),
    ACCOUNT_NOT_MATCH(HttpStatus.BAD_REQUEST, "계좌 번호가 일치하지 않습니다."), 
    FAIL_CHECK_AUTH(HttpStatus.INTERNAL_SERVER_ERROR, "인증이 실패했습니다");

    private final HttpStatus httpStatus;
    private final String message;
}
