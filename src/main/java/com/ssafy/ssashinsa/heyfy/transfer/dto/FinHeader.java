package com.ssafy.ssashinsa.heyfy.transfer.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/** SSAFY 금융망 공통 헤더 DTO */
@Getter @Setter
@Builder @NoArgsConstructor @AllArgsConstructor
public class FinHeader {
    private String apiName;
    private String transmissionDate; // yyyyMMdd
    private String transmissionTime; // HHmmss
    private String institutionCode;
    private String fintechAppNo;
    private String apiServiceCode;
    private String institutionTransactionUniqueNo; // yyyyMMdd+HHmmss+일렬번호 6자리
    private String apiKey;
    private String userKey;
}