package com.ssafy.ssashinsa.heyfy.exchange.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum ExchangeErrorCode implements ErrorCode {
    INVALID_PIN_NUMBER(HttpStatus.BAD_REQUEST, "Wrong pin number"),
    ACCOUNT_NOT_FOUND(HttpStatus.BAD_REQUEST, "Account not found"),
    INSUFFICIENT_BALANCE(HttpStatus.BAD_REQUEST, "Insufficient account balance"),
    EXCHANGE_MIN_UNIT(HttpStatus.BAD_REQUEST, "Minimum exchange unit is 10"),
    EXCHANGE_MIN_AMOUNT(HttpStatus.BAD_REQUEST, "Minimum exchange amount is 100 USD"),
    INVALID_TRANSACTION_AMOUNT(HttpStatus.BAD_REQUEST, "Invalid transaction amount"),
    FOREIGN_ACCOUNT_ONLY(HttpStatus.BAD_REQUEST, "Only foreign currency accounts are allowed"),
    KRW_ACCOUNT_ONLY(HttpStatus.BAD_REQUEST, "Only KRW accounts are allowed"),
    ALREADY_CANCELED(HttpStatus.BAD_REQUEST, "This reservation is already canceled"),
    ALREADY_COMPLETED(HttpStatus.BAD_REQUEST, "This reservation is already completed");

    private final HttpStatus httpStatus;
    private final String message;
}
