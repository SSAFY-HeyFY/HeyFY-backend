package com.ssafy.ssashinsa.heyfy.transfer.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
@AllArgsConstructor
public enum TransferErrorCode implements ErrorCode {
    INVALID_PIN_NUMBER(HttpStatus.BAD_REQUEST, "Wrong pin number"),
    INSUFFICIENT_BALANCE(HttpStatus.BAD_REQUEST, "Transaction failed due to insufficient account balance"), // A1014
    INVALID_TRANSACTION_AMOUNT(HttpStatus.BAD_REQUEST, "The transaction amount is invalid"),               // A1011
    INVALID_ACCOUNT_NUMBER(HttpStatus.BAD_REQUEST, "The account number is invalid"),                       // A1003
    EXCEEDED_TRANSACTION_SUMMARY_LENGTH(HttpStatus.BAD_REQUEST, "The transaction summary length has been exceeded"), // A1018
    ONLY_FOREIGN_CURRENCY_ACCOUNT(HttpStatus.BAD_REQUEST, "Only foreign currency accounts are allowed");   // A5005

    private final HttpStatus httpStatus;
    private final String message;
}
