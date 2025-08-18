package com.ssafy.ssashinsa.heyfy.transfer.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

public class CustomExceptions {

    /** 400 Bad Request - 사용자의 요청이 유효하지 않을 때 */
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public static class InvalidRequestException extends RuntimeException {
        public InvalidRequestException(String message) {
            super(message);
        }
    }

    /** 502 Bad Gateway - 외부 시스템 연동 실패 */
    @ResponseStatus(HttpStatus.BAD_GATEWAY)
    public static class ExternalApiCallException extends RuntimeException {
        public ExternalApiCallException(String message) {
            super(message);
        }
    }
}