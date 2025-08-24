package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member;

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
