package com.ssafy.ssashinsa.heyfy.account.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonResponseHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccountAuthCheckResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private AccountAuthCheckResponseRecDto REC;
}
