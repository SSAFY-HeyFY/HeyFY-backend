package com.ssafy.ssashinsa.heyfy.transfer.dto;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class FinRespHeader {
    private String responseCode;
    private String responseMessage;
    private String apiName;
    private String transmissionDate;
    private String transmissionTime;
    private String institutionCode;
    private String apiKey;
    private String apiServiceCode;
    private String institutionTransactionUniqueNo;
}
