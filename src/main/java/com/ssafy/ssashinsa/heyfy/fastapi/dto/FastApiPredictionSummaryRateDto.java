package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import lombok.Data;

@Data
public class FastApiPredictionSummaryRateDto {
    private final double percent;
    private final int days;

    // 문자열을 받아서 직접 파싱
    @JsonCreator
    public FastApiPredictionSummaryRateDto(String value) {
        // "+0.40% in 6 days" 같은 값이 들어온다고 가정
        String[] parts = value.replace("%", "").replace("+", "").split(" in ");
        this.percent = Double.parseDouble(parts[0]);
        this.days = Integer.parseInt(parts[1].replace(" days", ""));
    }
}