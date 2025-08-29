package com.ssafy.ssashinsa.heyfy.authentication.docs;

import com.ssafy.ssashinsa.heyfy.authentication.dto.TokenDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "액세스 토큰 갱신", description = "리프레시 토큰으로 만료된 액세스 토큰을 갱신합니다.(10초 이내 동일한 요청에 대해서는 같은 액세스 토큰, 리프레시 토큰을 반환합니다)")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "토큰 갱신 성공",
                content = @Content(schema = @Schema(implementation = TokenDto.class),
                        examples = @ExampleObject(
                                name = "토큰 갱신 성공 응답",
                                value = "{\"accessToken\":\"new_jwt_token_example\", \"refreshToken\":\"new_refresh_token_example\"}"
                        ))),
        @ApiResponse(responseCode = "400", description = "잘못된 토큰",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "유효하지 않은 리프레시 토큰",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"INVALID_REFRESH_TOKEN\", \"message\":\"유효하지 않은 리프레시 토큰입니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "토큰 쌍 불일치",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"TOKEN_PAIR_MISMATCH\", \"message\":\"액세스 토큰과 리프레시 토큰의 쌍이 올바르지 않습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "만료된 리프레시 토큰",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"EXPIRED_REFRESH_TOKEN\", \"message\":\"리프레쉬 토큰이 만료되었습니다.\"}"
                                )
                        })),
        @ApiResponse(responseCode = "401", description = "인증 실패 (토큰 누락)",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "토큰 누락",
                                        value = "{\"status\":401, \"httpError\":\"UNAUTHORIZED\", \"errorCode\":\"MISSING_ACCESS_TOKEN\", \"message\":\"액세스 토큰이 누락되었습니다.\"}"
                                )
                        })),
        @ApiResponse(responseCode = "409", description = "토큰 재발급 요청 중복",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "토큰 재발급 요청 중복",
                                value = "{\"status\":409, \"httpError\":\"CONFLICT\", \"errorCode\":\"TOKEN_REFRESH_IN_PROGRESS\", \"message\":\"토큰 재발급 요청이 이미 진행 중입니다.\"}"
                        )))
})
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER)

@Parameter(name = "RefreshToken", description = "JWT 리프레시 토큰", in = ParameterIn.HEADER, required = true)
public @interface AuthRefreshDocs {
}
