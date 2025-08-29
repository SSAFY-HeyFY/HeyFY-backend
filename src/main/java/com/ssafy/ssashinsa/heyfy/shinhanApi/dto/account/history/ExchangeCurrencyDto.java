package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeCurrencyDto {
    @JsonProperty("currency")
    private String currency;

    @JsonProperty("currencyName")
    private String currencyName;

    @JsonProperty("amount")
    private Double amount; // 📌 Double 타입으로 수정

    @JsonProperty("exchangeRate")
    private String exchangeRate; // 📌 Double 타입으로 수정
}
