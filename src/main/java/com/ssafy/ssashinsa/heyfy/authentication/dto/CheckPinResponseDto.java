package com.ssafy.ssashinsa.heyfy.authentication.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class CheckPinResponseDto {
    @JsonProperty("isCorrect")
    private boolean isCorrect;
    private String txnToken;
}
