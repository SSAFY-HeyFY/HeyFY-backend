package com.ssafy.ssashinsa.heyfy.shinhanApi.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanUserRequestDto {
    private String apiKey;
    private String userId;
}
