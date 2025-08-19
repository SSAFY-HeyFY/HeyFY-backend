package com.ssafy.ssashinsa.heyfy.register.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCreateDepositResponseHeaderDto {
    private String apiName;
    private String transmissionDate;
    private String transmissionTime;
    private String fintechAppNo;
    private String institutionCode;
    private String apiServiceCode;
    private String institutionTransactionUniqueNo;
    private String userKey;
    private String responseCode;
    private String responseMessage;
}