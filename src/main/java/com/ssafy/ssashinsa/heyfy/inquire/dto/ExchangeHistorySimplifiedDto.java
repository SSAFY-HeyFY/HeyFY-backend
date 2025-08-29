package com.ssafy.ssashinsa.heyfy.inquire.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeHistorySimplifiedDto {

    @JsonProperty("fromAccountNo")
    private String fromAccountNo;

    @JsonProperty("toAccountNo")
    private String toAccountNo;

    @JsonProperty("currency")
    private String currency;

    @JsonProperty("currencyName")
    private String currencyName;

    @JsonProperty("amount")
    private String amount;

    @JsonProperty("exchangeCurrency")
    private String exchangeCurrency;

    @JsonProperty("exchangeCurrencyName")
    private String exchangeCurrencyName;

    @JsonProperty("exchangeAmount")
    private String exchangeAmount;

    @JsonProperty("exchangeRate")
    private String exchangeRate;

    @JsonProperty("created")
    private String created;
}
