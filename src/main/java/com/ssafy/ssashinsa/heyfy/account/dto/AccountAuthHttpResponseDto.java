package com.ssafy.ssashinsa.heyfy.account.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AccountAuthHttpResponseDto {
    private String message;
    private String accountNo;
}