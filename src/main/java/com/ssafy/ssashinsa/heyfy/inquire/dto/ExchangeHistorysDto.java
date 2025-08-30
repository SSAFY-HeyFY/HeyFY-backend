package com.ssafy.ssashinsa.heyfy.inquire.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeHistorysDto {
    List<ExchangeHistorySimplifiedDto> exchangeHistorys;
}
