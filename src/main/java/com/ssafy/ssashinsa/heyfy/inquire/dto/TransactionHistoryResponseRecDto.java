package com.ssafy.ssashinsa.heyfy.inquire.dto;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class TransactionHistoryResponseRecDto {
    private String totalCount;
    private List<TransactionHistoryDto> list;
}