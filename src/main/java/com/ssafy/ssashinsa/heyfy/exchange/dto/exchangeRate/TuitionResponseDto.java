package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "등록금 납부일자 중 추천 dto")
public class TuitionResponseDto {
    private PeriodResponseDto period;
    @JsonFormat(shape = JsonFormat.Shape.STRING, pattern = "yyyyMMdd")
    private String recommendedDate;
    private String recommendationNote;
}

