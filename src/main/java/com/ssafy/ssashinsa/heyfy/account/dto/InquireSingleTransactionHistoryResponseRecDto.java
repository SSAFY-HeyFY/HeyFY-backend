package com.ssafy.ssashinsa.heyfy.account.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InquireSingleTransactionHistoryResponseRecDto {
    private String transactionUniqueNo;
    private String transactionDate;
    private String transactionTime;
    private String transactionType;
    private String transactionTypeName;
    private String transactionAccountNo;
    private double transactionBalance;
    private double transactionAfterBalance;
    private String transactionSummary;
    private String transactionMemo;
}
