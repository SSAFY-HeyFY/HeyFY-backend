package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RealTimeRateGroupDto {
    private RealTimeRateDto usd;
    private RealTimeRateDto cny;
    private RealTimeRateDto vnd;
}
