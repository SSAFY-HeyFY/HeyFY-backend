package com.ssafy.ssashinsa.heyfy.inquire.dto;

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
    private String dailyTransferLimit;
    private String oneTimeTransferLimit;
    private String accountBalance;
    private String lastTransactionDate;
    private String currency;
}