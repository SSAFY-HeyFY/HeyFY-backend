package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FastApiRateGraphDataDto {
    private LocalDate date;
    private double rate;
    @JsonProperty("is_prediction")
    private boolean isPrediction;
    private String modelName;
}

