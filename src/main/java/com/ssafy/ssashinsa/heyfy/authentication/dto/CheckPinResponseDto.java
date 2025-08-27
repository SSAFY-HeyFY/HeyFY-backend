package com.ssafy.ssashinsa.heyfy.authentication.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class CheckPinResponseDto {
    private boolean isCorrect;
    private String txnToken;
}
