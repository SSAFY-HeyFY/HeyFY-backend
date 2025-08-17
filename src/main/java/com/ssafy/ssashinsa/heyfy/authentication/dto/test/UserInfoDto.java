package com.ssafy.ssashinsa.heyfy.authentication.dto.test;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Schema
public class UserInfoDto {
    private String currentUsername;
    private String currentUserKey;
}