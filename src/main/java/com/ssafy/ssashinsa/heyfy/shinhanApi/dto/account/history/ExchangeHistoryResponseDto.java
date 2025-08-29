package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonResponseHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeHistoryResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto Header;

    @JsonProperty("REC")
    private List<ExchangeHistoryRecDto> REC;
}
