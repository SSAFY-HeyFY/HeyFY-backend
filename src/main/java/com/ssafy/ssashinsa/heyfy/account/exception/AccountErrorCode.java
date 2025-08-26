package com.ssafy.ssashinsa.heyfy.account.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

@Getter
@RequiredArgsConstructor
public enum AccountErrorCode implements ErrorCode {
    WITHDRAWAL_ACCOUNT_NOT_FOUND(HttpStatus.NOT_FOUND, "Account not found."),
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "API call failed.");

    private final HttpStatus httpStatus;
    private final String message;
}