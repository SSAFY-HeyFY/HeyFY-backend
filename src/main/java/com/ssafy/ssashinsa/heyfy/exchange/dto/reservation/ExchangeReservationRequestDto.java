package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeReservationRequestDto {
    private Double transactionBalance;
    private String currency;
    private String pinNumber;
    private Double baseExchangeRate;
}
