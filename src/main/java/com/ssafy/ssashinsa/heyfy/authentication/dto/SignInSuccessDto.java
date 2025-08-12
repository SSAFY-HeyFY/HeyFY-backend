package com.ssafy.ssashinsa.heyfy.authentication.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class SignInSuccessDto {
    private String accessToken;
}
