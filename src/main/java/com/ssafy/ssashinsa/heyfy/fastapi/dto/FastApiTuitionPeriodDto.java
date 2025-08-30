package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FastApiTuitionPeriodDto {
    @JsonProperty("highest_rate_date")
    private String recommendDate;
    @JsonProperty("highest_rate_value")
    private String value;
}
