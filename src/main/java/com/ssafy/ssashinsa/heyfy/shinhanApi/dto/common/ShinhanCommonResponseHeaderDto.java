package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCommonResponseHeaderDto {
    private String responseCode;
    private String responseMessage;
    private String apiName;
    private String transmissionDate;
    private String transmissionTime;
    private String institutionCode;
    private String apiKey;
    private String apiServiceCode;
    private String institutionTransactionUniqueNo;
    private String userKey;
}
