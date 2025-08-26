package com.ssafy.ssashinsa.heyfy.common.exception;

import org.springframework.http.HttpStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum CommonErrorCode implements ErrorCode {
    RESOURCE_NOT_FOUND(HttpStatus.NOT_FOUND, "The requested address was not found."),
    INVALID_FIELD(HttpStatus.BAD_REQUEST, "Invalid field."),
    INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "An error has occurred."),
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "User not found."),
    Shinhan_API_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "A banking system error occurred. Please try again later.");

    private final HttpStatus httpStatus;
    private final String message;
}