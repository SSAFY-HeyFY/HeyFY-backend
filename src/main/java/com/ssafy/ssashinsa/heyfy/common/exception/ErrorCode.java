package com.ssafy.ssashinsa.heyfy.common.exception;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;

public interface ErrorCode {
    HttpStatus getHttpStatus(); // HTTP 상태 코드 반환
    String getMessage();
    String name();
}
