package com.ssafy.ssashinsa.heyfy.authentication.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@Schema(description = "sid 응답 DTO")
public class SidDto {
    private String sid;
    @JsonProperty("isCorrect")
    private boolean isCorrect;
}
