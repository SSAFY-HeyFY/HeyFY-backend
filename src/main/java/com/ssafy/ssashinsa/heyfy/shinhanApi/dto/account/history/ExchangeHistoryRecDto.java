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
public class ExchangeHistoryRecDto {

    @JsonProperty("account")
    private AccountDto account;

    @JsonProperty("currency")
    private CurrencyDto currency;

    @JsonProperty("exchangeCurrency")
    private ExchangeCurrencyDto exchangeCurrency;

    @JsonProperty("created")
    private String created;
}