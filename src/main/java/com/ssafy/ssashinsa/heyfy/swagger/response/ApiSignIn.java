package com.ssafy.ssashinsa.heyfy.swagger.response;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInSuccessDto;
import com.ssafy.ssashinsa.heyfy.swagger.dto.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
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
@Operation(summary = "로그인", description = "아이디와 비밀번호로 로그인하고, JWT 토큰을 발급받습니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "로그인 성공",
                content = @Content(schema = @Schema(implementation = SignInSuccessDto.class),
                        examples = @ExampleObject(
                                name = "로그인 성공 응답",
                                value = "{\"accessToken\":\"eyJ...\", \"refreshToken\":\"eyJ...\"}"
                        ))),
        @ApiResponse(responseCode = "400", description = "로그인 실패",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "로그인 정보 불일치",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"LOGIN_FAILED\", \"message\":\"아이디 또는 비밀번호가 올바르지 않습니다.\"}"
                                )
                        })),
})
public @interface ApiSignIn {
}