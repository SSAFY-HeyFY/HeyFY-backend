package com.ssafy.ssashinsa.heyfy.shinhanApi.exception;

import lombok.Getter;

@Getter
public class ShinhanException extends RuntimeException {
    private final ShinhanErrorCode errorCode;

    public ShinhanException(ShinhanErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
    }
}
