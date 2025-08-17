package com.ssafy.ssashinsa.heyfy.swagger.response;

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
@Operation(summary = "보호된 엔드포인트", description = "JWT 토큰이 필요한 보호된 엔드포인트입니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "성공",
                content = @Content(schema = @Schema(example = "{\"message\":\"안녕하세요, 사용자이름님! JWT 토큰이 있어야만 접근 가능한 보호된 엔드포인트입니다.\"}"))),
        @ApiResponse(responseCode = "401", description = "인증 실패 (JWT 토큰 누락/만료/오류)",
                content = @Content(schema = @Schema(implementation = com.ssafy.ssashinsa.heyfy.swagger.dto.ErrorResponse.class))),
})
public @interface ApiProtected {
}