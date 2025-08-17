package com.ssafy.ssashinsa.heyfy.authentication.docs;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignUpSuccessDto;
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
@Operation(summary = "회원가입", description = "새로운 사용자를 등록합니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "회원가입 성공",
                content = @Content(schema = @Schema(implementation = SignUpSuccessDto.class),
                        examples = @ExampleObject(
                                name = "회원가입 성공 응답",
                                value = "{\"message\":\"회원가입이 성공적으로 완료되었습니다.\", \"username\":\"testuser\"}"
                        ))),
        @ApiResponse(responseCode = "400", description = "회원가입 실패",
                content = @Content(schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "유효성 검사 실패",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"INVALID_FIELD\", \"message\":\"비밀번호는 8자 이상이어야 합니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "중복된 아이디",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"EXIST_USER_NAME\", \"message\":\"이미 존재하는 아이디입니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "중복된 이메일",
                                        value = "{\"status\":400, \"httpError\":\"BAD_REQUEST\", \"errorCode\":\"EXIST_EMAIL\", \"message\":\"이미 가입된 이메일입니다.\"}"
                                )
                        })),
})
public @interface AuthSignUpDocs {
}