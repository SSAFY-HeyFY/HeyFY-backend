package com.ssafy.ssashinsa.heyfy.authentication.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@Schema(description = "로그인 요청 DTO")
public class SignInDto {
    private String username;
    private String password;
}
