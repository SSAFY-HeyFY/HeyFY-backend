package com.ssafy.ssashinsa.heyfy.shinhanApi.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanErrorResponseDto {
    private String responseCode;
    private String responseMessage;
}
