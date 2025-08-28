package com.ssafy.ssashinsa.heyfy.authentication.docs;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SidDto;
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
@Operation(summary = "SID 재발급", description = "유효한 액세스 토큰과 2차 비밀번호를 통해 새로운 SID를 발급합니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "SID 재발급 성공",
                content = @Content(schema = @Schema(implementation = SidDto.class),
                        examples = @ExampleObject(
                                name = "SID 재발급 성공 응답",
                                value = "{\"sid\": \"6d69e0b1-024d-4aab-805b-ae62cb8f6419\"}"
                        ))),
        @ApiResponse(responseCode = "400", description = "잘못된 요청 또는 비밀번호",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "2차 비밀번호 불일치",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"PIN_NUMBER_MISMATCH\", \"message\":\"2차 비밀번호가 일치하지 않습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "잘못된 요청 본문",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"INVALID_REQUEST_BODY\", \"message\":\"요청 본문이 없거나 형식이 유효하지 않습니다.\"}"
                                )
                        })),
        @ApiResponse(responseCode = "401", description = "인증 실패 (JWT 토큰 누락 또는 만료)",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "유효하지 않은 액세스 토큰",
                                        value = "{\"status\":401, \"httpError\":\"UNAUTHORIZED\", \"errorCode\":\"INVALID_ACCESS_TOKEN\", \"message\":\"유효하지 않은 액세스 토큰입니다.\"}"
                                )
                        }))
})
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER)

public @interface SidRefreshDocs {
}