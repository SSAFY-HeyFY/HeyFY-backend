package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import lombok.Data;

@Data
public class ShinhanInquireDepositResponseRecDto {
    private String bankCode;
    private String bankName;
    private String userName;
    private String accountNo;
    private String accountName;
    private String accountTypeCode;
    private String accountTypeName;
    private String accountCreatedDate;
    private String accountExpiryDate;
    private double dailyTransferLimit;
    private double oneTimeTransferLimit;
    private double accountBalance;
    private String lastTransactionDate;
    private String currency;
}