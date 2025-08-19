package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCommonRequestHeaderDto {
    private String apiName;
    private String transmissionDate;
    private String transmissionTime;
    private String fintechAppNo;
    private String institutionCode;
    private String apiKey;
    private String apiServiceCode;
    private String institutionTransactionUniqueNo;
    private String userKey;
}
