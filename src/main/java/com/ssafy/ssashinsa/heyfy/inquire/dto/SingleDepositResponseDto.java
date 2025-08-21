package com.ssafy.ssashinsa.heyfy.inquire.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class SingleDepositResponseDto {
    private String bankName;
    private String userName;
    private String accountNo;
    private String accountName;
    private String accountBalance;
    private String currency;
}
