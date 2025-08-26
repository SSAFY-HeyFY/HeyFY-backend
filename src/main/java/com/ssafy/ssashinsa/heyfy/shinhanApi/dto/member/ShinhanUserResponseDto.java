package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanUserResponseDto {
    private String userId;
    private String userName;
    private String userKey;
    private String institutionCode;
}
