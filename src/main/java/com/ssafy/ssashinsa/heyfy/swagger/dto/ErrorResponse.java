package com.ssafy.ssashinsa.heyfy.swagger.dto;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.Instant;

@Data
@Schema(description = "에러 공통 응답")
public class ErrorResponse {

    @Schema(description = "에러 코드", example = "400")
    private int code;

    @Schema(description = "에러 메시지", example = "Api request body invalid")
    private String message;

    @Schema(description = "발생 시각(UTC)", example = "2025-08-13T14:07:15Z")
    private Instant timestamp;

    public ErrorResponse() {}

    public ErrorResponse(int code, String message) {
        this.code = code;
        this.message = message;
        this.timestamp = Instant.now();
    }

    public static ErrorResponse of(int code, String message) {
        return new ErrorResponse(code, message);
    }

    // Getter & Setter
}