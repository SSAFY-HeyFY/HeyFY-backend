package com.ssafy.ssashinsa.heyfy.common.exception;

import org.springframework.http.HttpStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum CommonErrorCode implements ErrorCode {
    RESOURCE_NOT_FOUND(HttpStatus.NOT_FOUND, "요청한 주소를 찾을 수 없습니다."),
    INVALID_FIELD(HttpStatus.BAD_REQUEST, "잘못된 필드입니다."),
    INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "에러가 발생했습니다"),
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "유저를 찾을 수 없습니다."),
    Shinhan_API_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "은행 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
    INVALID_REQUEST_BODY(HttpStatus.BAD_REQUEST, "요청 본문이 없거나 형식이 유효하지 않습니다.");


    private final HttpStatus httpStatus;
    private final String message;
}
