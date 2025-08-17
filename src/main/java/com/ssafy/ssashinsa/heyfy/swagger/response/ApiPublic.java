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
@Operation(summary = "공개 엔드포인트", description = "인증 없이 접근 가능한 공개 엔드포인트입니다.")
@ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "성공",
                content = @Content(schema = @Schema(example = "{\"message\":\"여기는 인증 없이 접근할 수 있는 공개 엔드포인트입니다.\"}"))),
})
public @interface ApiPublic {
}