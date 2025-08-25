package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FastApiRealTimeRateDto {
    private String updatedAt;
    private String currency;
    private Double rate;
    private String changeDirection;
    private Double changeAbs;
    private Double changePct;
    private Double cashBuy;
    private Double cashSell;
    private Double wireSend;
    private Double wireReceive;
    private String provider;
}

