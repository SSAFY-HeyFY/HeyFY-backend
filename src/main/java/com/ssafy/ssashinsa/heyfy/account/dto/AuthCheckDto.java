package com.ssafy.ssashinsa.heyfy.account.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class AuthCheckDto {
    private String accountNo;
    private String authCode;
}
