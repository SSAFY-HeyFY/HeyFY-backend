package com.ssafy.ssashinsa.heyfy.exchange.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TuitionDto {
    private PeriodDto period;
    private String recommendedDate;
    private String recommendationNote;
}

