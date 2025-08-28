package com.ssafy.ssashinsa.heyfy.authentication.docs;

import com.ssafy.ssashinsa.heyfy.authentication.dto.CheckPinResponseDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.PinNumberDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "PIN 번호 검증", description = "2차 인증을 위해 PIN 번호의 일치 여부를 확인하고, 성공 시 트랜잭션 토큰을 발급합니다. PIN 번호가 틀려도 HTTP 상태 코드는 200이며, 응답 본문의 'correct' 필드로 결과를 알려줍니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "PIN 번호 확인 결과 반환",
                content = @Content(schema = @Schema(implementation = CheckPinResponseDto.class),
                        examples = {
                                @ExampleObject(
                                        name = "PIN 번호 확인 성공",
                                        value = "{\"txnToken\":\"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3Nzc3Nzc3IiwianRpIjoiZGI4NDU5NmQtMzJhNi00MDA5LTgwNGMtYzI0ODVlYjJiZmZhIiwiaWF0IjoxNzU2MjU0NTc5LCJleHAiOjE3NTYyNTUxNzl9.NJwdTUSiZAU3if-H5gplEkFVPo2YCcSXo0-TLnKUsBY\", \"correct\":true}"
                                ),
                                @ExampleObject(
                                        name = "PIN 번호 확인 실패",
                                        value = "{\"txnToken\": null, \"correct\": false}"
                                )
                        })),
        @ApiResponse(responseCode = "401", description = "인증 실패 (액세스 토큰 누락)",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "액세스 토큰 누락",
                                value = "{\"status\":401, \"httpError\":\"UNAUTHORIZED\", \"errorCode\":\"MISSING_ACCESS_TOKEN\", \"message\":\"액세스 토큰이 누락되었습니다.\"}"
                        )))
})
@RequestBody(
        description = "PIN 번호",
        required = true,
        content = @Content(
                schema = @Schema(implementation = PinNumberDto.class),
                examples = @ExampleObject(
                        name = "PIN 번호 요청 예시",
                        value = "{\"pinNumber\":\"1234\"}"
                )
        )
)
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER)
@Parameter(name = "sid", description = "세션 ID", in = ParameterIn.HEADER)
public @interface AuthCheckPinDocs {
}