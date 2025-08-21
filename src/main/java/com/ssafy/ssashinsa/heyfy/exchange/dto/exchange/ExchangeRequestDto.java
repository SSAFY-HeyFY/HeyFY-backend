package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ExchangeRequestDto {
    private String depositAccountNo;
    private String depositAccountCurrency;
    private String withdrawalAccountNo;
    private String withdrawalAccountCurrency;
    private Long transactionBalance;
}

