package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.exchange;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class ShinhanExchangeResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanExchangeResponseRecDto REC;
}
