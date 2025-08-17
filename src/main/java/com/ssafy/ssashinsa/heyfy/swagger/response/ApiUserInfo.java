package com.ssafy.ssashinsa.heyfy.swagger.response;

import com.ssafy.ssashinsa.heyfy.authentication.dto.test.UserInfoDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "사용자 정보 조회", description = "현재 로그인된 사용자의 정보를 조회합니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "성공",
                content = @Content(schema = @Schema(implementation = UserInfoDto.class))),
        @ApiResponse(responseCode = "401", description = "인증 실패 (JWT 토큰 누락/만료/오류)",
                content = @Content(schema = @Schema(implementation = com.ssafy.ssashinsa.heyfy.swagger.dto.ErrorResponse.class))),
})
public @interface ApiUserInfo {
}