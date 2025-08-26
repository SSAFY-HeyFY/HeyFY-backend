package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TransferResponseDto {
    private String transactionUniqueNo;
    private String accountNo;
    private String transactionDate;     // yyyyMMdd
    private String transactionType;     // "1","2" 등
    private String transactionTypeName; // 입금(이체) / 출금(이체)
    private String transactionAccountNo;// 상대 계좌번호
}
