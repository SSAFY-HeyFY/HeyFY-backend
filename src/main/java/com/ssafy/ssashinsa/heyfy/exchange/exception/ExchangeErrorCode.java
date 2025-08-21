package com.ssafy.ssashinsa.heyfy.exchange.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ExchangeErrorCode implements ErrorCode {
    ACCOUNT_NOT_FOUND(HttpStatus.BAD_REQUEST, "Account not found");
    private final HttpStatus httpStatus;
    private final String message;
}
