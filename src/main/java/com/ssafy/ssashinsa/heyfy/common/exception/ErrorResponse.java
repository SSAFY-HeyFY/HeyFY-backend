package com.ssafy.ssashinsa.heyfy.common.exception;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.springframework.http.ResponseEntity;

@Getter
@Setter
@Builder
@AllArgsConstructor
@Schema(description = "공통 에러 응답 DTO")
public class ErrorResponse {
    private final int status;
    private final String httpError;
    private final String errorCode;
    private final String message;
    public static ResponseEntity<ErrorResponse> responseEntity(ErrorCode errorCode) {
        return ResponseEntity
                .status(errorCode.getHttpStatus())
                .body(ErrorResponse.builder()
                        .status(errorCode.getHttpStatus().value())
                        .httpError(errorCode.getHttpStatus().name())
                        .errorCode(errorCode.name())
                        .message(errorCode.getMessage())
                        .build()
                );
    }
    public static ResponseEntity<ErrorResponse> responseEntity(ErrorCode errorCode, String message) {
        return ResponseEntity
                .status(errorCode.getHttpStatus())
                .body(ErrorResponse.builder()
                        .status(errorCode.getHttpStatus().value())
                        .httpError(errorCode.getHttpStatus().name())
                        .errorCode(errorCode.name())
                        .message(message)
                        .build()
                );
    }
}
