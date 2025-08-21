package com.ssafy.ssashinsa.heyfy.inquire.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ForeignSingleDepositResponseDto {
    private String bankName;
    private String userName;
    private String accountNo;
    private String accountName;
    private String accountBalance; // String 타입
    private String currency;
}