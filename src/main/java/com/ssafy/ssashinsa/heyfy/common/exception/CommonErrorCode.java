package com.ssafy.ssashinsa.heyfy.common.exception;

import org.springframework.http.HttpStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum CommonErrorCode implements ErrorCode {
    RESOURCE_NOT_FOUND(HttpStatus.NOT_FOUND, "리소스를 찾을 수 없습니다."),
    INVALID_FIELD(HttpStatus.BAD_REQUEST, "잘못된 필드입니다."),
    INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "에러가 발생했습니다"),
    USER_NOT_FOUND(HttpStatus.INTERNAL_SERVER_ERROR, "유저를 찾을 수 없습니다."),;

    private final HttpStatus httpStatus;
    private final String message;
}
