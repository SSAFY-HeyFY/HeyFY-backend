package com.ssafy.ssashinsa.heyfy.register.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ShinhanRegisterApiErrorCode implements ErrorCode {
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "Shinhan API call failed."),
    MISSING_USER_KEY(HttpStatus.UNAUTHORIZED, "User key is missing."),
    USER_NOT_FOUND(HttpStatus.BAD_REQUEST, "User not found."),
    RESPONSE_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "Shinhan API response error."),
    ACCOUNT_ALREADY_EXISTS(HttpStatus.BAD_REQUEST, "User already has an account."),
    ACCOUNT_NOT_FOUND(HttpStatus.BAD_REQUEST, "Account associated with the user was not found."),
    ACCOUNT_NOT_MATCH(HttpStatus.BAD_REQUEST, "Account number does not match."),
    FAIL_CHECK_AUTH(HttpStatus.INTERNAL_SERVER_ERROR, "Authentication failed.");

    private final HttpStatus httpStatus;
    private final String message;
}