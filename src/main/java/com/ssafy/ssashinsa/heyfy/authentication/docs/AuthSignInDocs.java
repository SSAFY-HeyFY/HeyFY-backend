package com.ssafy.ssashinsa.heyfy.authentication.docs;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInSuccessDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
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
                                // 💡 요청하신 JSON 데이터를 그대로 삽입
                                value = "{\n" +
                                        "  \"accessToken\": \"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3Nzc3Nzc3Iiwicm9sZXMiOiIiLCJqdGkiOiJkODNmMTcxNC0zZjNlLTQzYjEtODhkOS0zNjhmYjcwMzhhNzQiLCJpYXQiOjE3NTYxOTQ0NjQsImV4cCI6MTc1NjE5ODA2NH0.WWnpJAxA8N2E8g3YC8QMJBQs-UADfEnNaJBiL3B_Xr0\",\n" +
                                        "  \"refreshToken\": \"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3Nzc3Nzc3IiwianRpIjoiZDgzZjE3MTQtM2YzZS00M2IxLTg4ZDktMzY4ZmI3MDM4YTc0IiwiaWF0IjoxNzU2MTk0NDY0LCJleHAiOjE3NTY3OTkyNjR9.Skw1n1-VjA1NZcYagrRF9gYvYjVVJ-8vvqs3yHbU2cE\",\n" +
                                        "  \"sid\": \"959f77dc-c708-46d0-a9c1-abd5b4b9d011\"\n" +
                                        "}"
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
public @interface AuthSignInDocs {
}






