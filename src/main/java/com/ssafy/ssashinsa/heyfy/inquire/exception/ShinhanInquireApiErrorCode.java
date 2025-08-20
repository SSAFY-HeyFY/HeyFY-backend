package com.ssafy.ssashinsa.heyfy.inquire.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ShinhanInquireApiErrorCode implements ErrorCode {
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "신한API 호출 에러"),
    MISSING_USER_KEY(HttpStatus.BAD_REQUEST, "유저키가 누락되었습니다."),
    USER_NOT_FOUND(HttpStatus.INTERNAL_SERVER_ERROR, "유저를 찾을 수 없습니다."),;

    private final HttpStatus httpStatus;
    private final String message;
}
