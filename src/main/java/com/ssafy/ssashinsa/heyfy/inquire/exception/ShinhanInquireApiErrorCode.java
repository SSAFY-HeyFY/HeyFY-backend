package com.ssafy.ssashinsa.heyfy.inquire.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ShinhanInquireApiErrorCode implements ErrorCode {
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "Shinhan API call failed."),
    MISSING_USER_KEY(HttpStatus.BAD_REQUEST, "User key is missing."),
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "User not found.");

    private final HttpStatus httpStatus;
    private final String message;
}