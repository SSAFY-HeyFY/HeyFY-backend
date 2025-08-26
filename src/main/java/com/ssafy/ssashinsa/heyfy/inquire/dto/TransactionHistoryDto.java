package com.ssafy.ssashinsa.heyfy.inquire.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class TransactionHistoryDto {
    private String transactionUniqueNo;
    private String transactionDate;
    private String transactionTime;
    private String transactionType;
    private String transactionTypeName;
    private String transactionAccountNo;
    private String transactionBalance;
    private String transactionAfterBalance;
    private String transactionSummary;
    private String transactionMemo;
}
