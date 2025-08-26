package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import lombok.Data;

@Data
public class ShinhanInquireDemandDepositAccountBalanceResponseRecDto {
    private String bankCode;
    private String userName;
    private String accountNo;
    private Double accountBalance;
    private String accountCreatedDate;
    private String accountExpiryDate;
    private String lastTransactionDate;
    private String currency;
}
