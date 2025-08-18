package com.ssafy.ssashinsa.heyfy.transfer.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TransferRequestBody {
    @JsonProperty("Header")
    private FinHeader Header;

    private String depositAccountNo;              // 입금 계좌번호(16)
    private String depositTransactionSummary;     // 선택
    private String transactionBalance;            // 금액(문자열)
    private String withdrawalAccountNo;           // 출금 계좌번호(16)
    private String withdrawalTransactionSummary;  // 선택
}
