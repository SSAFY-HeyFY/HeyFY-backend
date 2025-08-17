package com.ssafy.ssashinsa.heyfy.authentication.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Schema(description = "토큰 응답 DTO")
public class TokenDto {
    private String accessToken;
    private String refreshToken;
}
