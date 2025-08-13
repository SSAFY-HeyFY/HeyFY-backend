package com.ssafy.ssashinsa.heyfy.authentication.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class SignInDto {
    private String username;
    private String password;
}
