package com.ssafy.ssashinsa.heyfy.register.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AccountCreationResponseDto {
    private String message;
    private String accountNo;
    private String currency;
}
