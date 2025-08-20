package com.ssafy.ssashinsa.heyfy.account.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

@Getter
@RequiredArgsConstructor
public enum AccountErrorCode implements ErrorCode {
    WITHDRAWAL_ACCOUNT_NOT_FOUND(HttpStatus.NOT_FOUND, "계좌를 찾을 수 없습니다."),
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "API 호출에 실패했습니다.");

    private final HttpStatus httpStatus;
    private final String message;
}