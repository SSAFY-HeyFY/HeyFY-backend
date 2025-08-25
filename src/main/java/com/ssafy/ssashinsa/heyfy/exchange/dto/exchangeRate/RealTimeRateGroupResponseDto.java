package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RealTimeRateGroupResponseDto {
    private RealTimeRateResponseDto usd;
    private RealTimeRateResponseDto cny;
    private RealTimeRateResponseDto vnd;
}
