package com.ssafy.ssashinsa.heyfy.transfer.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter @Setter
public class TransferResponseBody {

    @JsonProperty("Header")
    private FinRespHeader Header;

    @JsonProperty("REC")
    private List<Rec> REC;

    @Getter @Setter
    public static class Rec {
        private String transactionUniqueNo;
        private String accountNo;
        private String transactionDate;     // yyyyMMdd
        private String transactionType;     // "1","2" 등
        private String transactionTypeName; // 입금(이체) / 출금(이체)
        private String transactionAccountNo;// 상대 계좌번호
    }
}
