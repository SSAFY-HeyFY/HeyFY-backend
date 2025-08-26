package com.ssafy.ssashinsa.heyfy.fcm.docs;

import com.ssafy.ssashinsa.heyfy.fcm.dto.FcmTokenRequest;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.*;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Operation(
        summary = "FCM 토큰 등록 (로그인 필요)",
        description = "로그인된 사용자의 기기 FCM 토큰을 등록합니다.",
        security = { @SecurityRequirement(name = "Authorization") } // OpenApiConfig에서 정의한 보안 스키마명
)
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "등록 성공"),
        @ApiResponse(responseCode = "400", description = "필수 정보 누락"),
        @ApiResponse(responseCode = "404", description = "사용자 없음"),
        @ApiResponse(responseCode = "500", description = "서버 에러")
})
@RequestBody(
        description = "등록할 기기의 FCM 토큰",
        required = true,
        content = @Content(
                schema = @Schema(implementation = FcmTokenRequest.class),
                examples = @ExampleObject(value = "{ \"fcmToken\": \"fH7Zc...abcd1234\" }")
        )
)
public @interface FcmRegisterDocs {}
