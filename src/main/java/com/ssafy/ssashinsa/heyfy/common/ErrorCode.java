package com.ssafy.ssashinsa.heyfy.common;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ErrorCode {
    INVALID_FIELD(HttpStatus.BAD_REQUEST, "잘못된 필드입니다."),
    UNAUTHORIZED(HttpStatus.UNAUTHORIZED,"인증되지 않은 사용자입니다");
    private final HttpStatus httpStatus;
    private final String message;


}
