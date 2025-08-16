package com.ssafy.ssashinsa.heyfy.swagger.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "성공 기본 응답")
public class ResultSuccessResponseDto {

    @Schema(description = "결과 코드", example = "200")
    private int code;

    @Schema(description = "메시지", example = "OK")
    private String message;

    public ResultSuccessResponseDto() {}

    public ResultSuccessResponseDto(int code, String message) {
        this.code = code;
        this.message = message;
    }

    public static ResultSuccessResponseDto ok() {
        return new ResultSuccessResponseDto(200, "OK");
    }

    // Getter & Setter
}
